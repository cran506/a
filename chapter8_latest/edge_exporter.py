"""Auditable magnitude pruning and integer linear inference for SAC actors.

Only the deterministic actor mean is exported. Input channel balancing, input
quantization and final tanh use floating point; dense products, bias accumulation,
ReLU, and requantization use integers. This is a reference NumPy kernel, not an
MCU timing claim. Channel balancing is an algebraically equivalent affine prefix
folded into the first layer, not an additional learned control policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np


def _affine_params(values):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Quantization calibration must be finite")
    lo = min(0.0, float(a.min(initial=0.0)))
    hi = max(0.0, float(a.max(initial=0.0)))
    if hi == lo:
        return 1.0, 0
    scale = (hi - lo) / 255.0
    zero = int(np.clip(np.rint(-128.0 - lo / scale), -128, 127))
    return scale, zero


def _quantize(values, scale, zero):
    return np.clip(np.rint(np.asarray(values) / scale) + zero, -128, 127).astype(np.int8)


def magnitude_prune(weights, ratio):
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("prune_ratio must be in [0, 1]")
    w = np.array(weights, dtype=np.float32, copy=True)
    # A stable rank handles equal-magnitude ties deterministically; a percentile
    # alone can remove 0% or 100% when many weights equal the threshold.
    count = int(np.floor(ratio * w.size))
    flat = w.reshape(-1)
    mask = np.ones(flat.size, dtype=bool)
    mask[np.argsort(np.abs(flat), kind="stable")[:count]] = False
    flat[~mask] = 0.0
    return w, mask.reshape(w.shape)


@dataclass
class IntegerLayer:
    weights: np.ndarray
    bias: np.ndarray
    input_scale: float
    input_zero: int
    weight_scale: float
    weight_zero: int
    output_scale: float
    output_zero: int
    multiplier: int
    shift: int
    relu: bool

    def forward(self, activation):
        # Subtract zeros only AFTER casting; int8 subtraction would wrap.
        x = np.asarray(activation, dtype=np.int32) - self.input_zero
        w = self.weights.astype(np.int32) - self.weight_zero
        accum = x @ w.T + self.bias
        # int64 intermediate avoids fixed-point multiplier overflow. Magnitudes
        # are bounded at export time; no float GEMM is hidden in this path.
        scaled = accum.astype(np.int64) * np.int64(self.multiplier)
        denom = 1 << self.shift
        rounded = np.sign(scaled) * ((np.abs(scaled) + denom // 2) // denom)
        output = rounded + self.output_zero
        if self.relu:
            output = np.maximum(output, self.output_zero)
        return np.clip(output, -128, 127).astype(np.int8)


class EdgeLightweightExporter:
    """FP32, masked-pruned FP32, and INT8 references sharing one actor topology."""

    @staticmethod
    def prune_and_quantize_weights(W_fp32, prune_ratio=0.70):
        pruned, _ = magnitude_prune(W_fp32, prune_ratio)
        scale, zero = _affine_params(pruned)
        quant = _quantize(pruned, scale, zero)
        dequant = scale * (quant.astype(np.float32) - zero)
        sparsity = 100.0 * float(np.mean(pruned == 0.0))
        return dequant.astype(np.float32), quant, scale, zero, sparsity

    def __init__(self, weights, biases, calibration_obs, prune_ratio=.70,
                 pruned_weights=None, pruned_biases=None, balance_inputs=True,
                 balance_mode="smoothquant", smoothquant_alpha=.5):
        if len(weights) != len(biases) or not weights:
            raise ValueError("Nonempty matching weights/biases required")
        self.weights = [np.asarray(w, dtype=np.float32).copy() for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32).copy() for b in biases]
        self.masks = [magnitude_prune(w, prune_ratio)[1] for w in self.weights]
        self.pruned_weights = [w * mask for w, mask in zip(self.weights, self.masks)]
        self.pruned_biases = [b.copy() for b in self.biases]
        if pruned_weights is not None:
            self.pruned_weights = [np.asarray(w, dtype=np.float32).copy() for w in pruned_weights]
            self.pruned_biases = [np.asarray(b, dtype=np.float32).copy() for b in pruned_biases]
            if len(self.pruned_weights) != len(self.weights):
                raise ValueError("Student topology mismatch")
            for student, original, mask in zip(self.pruned_weights, self.weights, self.masks):
                if student.shape != original.shape or np.any(student[~mask] != 0.0):
                    raise ValueError("Student changed the fixed pruning mask")
        cal = np.asarray(calibration_obs, dtype=np.float32)
        if cal.ndim != 2 or cal.shape[1] != self.weights[0].shape[1] or len(cal) == 0:
            raise ValueError("calibration_obs must have shape (N, actor_input_dim)")
        if not np.isfinite(cal).all():
            raise ValueError("Nonfinite observation")
        self.prune_ratio = float(prune_ratio)
        self.calibration_count = len(cal)
        self.balance_inputs = bool(balance_inputs)
        if balance_mode not in {"smoothquant", "mean_std"}:
            raise ValueError("balance_mode must be smoothquant or mean_std")
        if not np.isfinite(smoothquant_alpha) or not 0. <= smoothquant_alpha <= 1.:
            raise ValueError("smoothquant_alpha must be finite and in [0,1]")
        self.balance_mode = balance_mode if self.balance_inputs else "none"
        self.smoothquant_alpha = float(smoothquant_alpha)
        self.input_balancing_floor = 1e-5
        self.integer_layers = self._build_integer_layers(cal)

    @classmethod
    def from_sb3(cls, model, calibration_obs, prune_ratio=.70, **kwargs):
        actor = model.actor if hasattr(model, "actor") else model.policy.actor
        weights, biases = cls.extract_actor(actor)
        return cls(weights, biases, calibration_obs, prune_ratio=prune_ratio, **kwargs)

    @staticmethod
    def extract_actor(actor):
        import torch.nn as nn
        layers = []
        for module in actor.latent_pi:
            if isinstance(module, nn.Linear):
                layers.append(module)
            elif not isinstance(module, nn.ReLU):
                raise ValueError("Integer reference supports Linear/ReLU actor mean only")
        if not isinstance(actor.mu, nn.Linear):
            raise ValueError("Expected SB3 SAC linear mean head")
        layers.append(actor.mu)
        weights = [m.weight.detach().cpu().numpy().astype(np.float32).copy() for m in layers]
        biases = [m.bias.detach().cpu().numpy().astype(np.float32).copy() for m in layers]
        return weights, biases

    @staticmethod
    def _float_forward(obs, weights, biases):
        x = np.asarray(obs, dtype=np.float32)
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = x @ w.T + b
            if i != len(weights) - 1:
                x = np.maximum(x, 0.0)
        return np.tanh(x)

    def _integer_float_parameters(self):
        """Equivalent sparse parameters for z=(obs-center)/channel_scale.

        Fold only after pruning/distillation. Multiplication by a nonzero
        channel scale preserves every existing zero; re-ranking the folded
        weights would change the selected pruning mask and the controller.
        """
        weights = [w.copy() for w in self.pruned_weights]
        biases = [b.copy() for b in self.pruned_biases]
        weights[0] = self.pruned_weights[0] * self.input_channel_scale[None, :]
        biases[0] = self.pruned_biases[0] + self.pruned_weights[0] @ self.input_center
        return weights, biases

    def _balanced_input(self, obs):
        return (np.asarray(obs, dtype=np.float32) - self.input_center) / self.input_channel_scale

    def _build_integer_layers(self, cal):
        cal = np.asarray(cal, dtype=np.float32)
        if cal.ndim != 2 or cal.shape[1] != self.weights[0].shape[1] or len(cal) == 0:
            raise ValueError("calibration_obs must have shape (N, actor_input_dim)")
        if not np.isfinite(cal).all():
            raise ValueError("Nonfinite observation")
        self.calibration_count = len(cal)
        if self.balance_inputs:
            self.input_center = cal.mean(axis=0, dtype=np.float64).astype(np.float32)
            if self.balance_mode == "mean_std":
                deviations = cal.std(axis=0, dtype=np.float64).astype(np.float32)
                scales = deviations
                active_channels = deviations > self.input_balancing_floor
            else:
                # SmoothQuant-style diagonal equilibration balances the
                # input and weight dynamic ranges before per-tensor INT8.
                # Statistics use calibration data, never the test trajectory.
                # An all-zero weight column has a finite denominator floor;
                # constant/near-constant inputs retain scale 1 and are kept.
                xmax = np.max(np.abs(cal-self.input_center), axis=0)
                wmax = np.max(np.abs(self.pruned_weights[0]), axis=0)
                eps = self.input_balancing_floor
                alpha = self.smoothquant_alpha
                scales = np.maximum(xmax, eps)**alpha / np.maximum(wmax, eps)**(1.-alpha)
                active_channels = xmax > eps
            self.input_channel_scale = np.where(active_channels, scales, 1.).astype(np.float32)
        else:
            self.input_center = np.zeros(cal.shape[1], dtype=np.float32)
            self.input_channel_scale = np.ones(cal.shape[1], dtype=np.float32)
        weights, biases = self._integer_float_parameters()
        layers = []
        x = self._balanced_input(cal)
        sx, zx = _affine_params(x)
        qx = _quantize(x, sx, zx)
        for i, (w, b) in enumerate(zip(weights, biases)):
            sw, zw = _affine_params(w)
            qw = _quantize(w, sw, zw)
            relu = i < len(self.weights) - 1
            # Calibration propagates the actual previous quantized activation,
            # so accumulated quantization effects enter later layer ranges.
            x = sx * (qx.astype(np.float32) - zx) @ w.T + b
            if relu:
                x = np.maximum(x, 0.0)
            sy, zy = _affine_params(x)
            raw_bias = np.rint(b / (sx * sw))
            if np.max(np.abs(raw_bias), initial=0.0) >= 2**31 - 1:
                raise ValueError("INT32 bias overflow; calibration/weight scale too small")
            qb = raw_bias.astype(np.int32)
            max_acc = w.shape[1] * 255 * 255 + np.max(np.abs(qb.astype(np.int64)), initial=0)
            if max_acc >= 2**31 - 1:
                raise ValueError("INT32 dot-product overflow")
            ratio = sx * sw / sy
            # Largest integer precision that still guarantees int64 rescale.
            shift = 30
            while shift > 0 and max_acc * max(1, round(ratio * (1 << shift))) >= 2**62:
                shift -= 1
            multiplier = int(round(ratio * (1 << shift)))
            if multiplier == 0:
                raise ValueError("Fixed-point multiplier underflow")
            layer = IntegerLayer(qw, qb, sx, zx, sw, zw, sy, zy,
                                 multiplier, shift, relu)
            layers.append(layer)
            qx = layer.forward(qx)
            sx, zx = sy, zy
        return layers

    def action(self, obs, variant="int8"):
        obs = np.asarray(obs, dtype=np.float32)
        if not np.isfinite(obs).all():
            raise ValueError("Nonfinite observation")
        if variant == "fp32":
            return self._float_forward(obs, self.weights, self.biases)
        if variant == "pruned_fp32":
            return self._float_forward(obs, self.pruned_weights, self.pruned_biases)
        if variant != "int8":
            raise ValueError("variant must be fp32, pruned_fp32, or int8")
        first = self.integer_layers[0]
        q = _quantize(self._balanced_input(obs), first.input_scale, first.input_zero)
        for layer in self.integer_layers:
            q = layer.forward(q)
        last = self.integer_layers[-1]
        return np.tanh(last.output_scale * (q.astype(np.float32) - last.output_zero))

    def distill_pruned(self, calibration_obs, steps=1000, batch_size=256,
                       learning_rate=1e-3, seed=2026):
        """Fit a fixed-mask sparse student to the trained FP32 actor actions.

        This post-training compression step uses teacher labels only. It is
        reported separately from environment interaction / SAC training.
        """
        import torch
        from torch import nn
        torch.manual_seed(seed)
        observations = torch.as_tensor(np.asarray(calibration_obs, dtype=np.float32))
        def build(weights, biases):
            modules = []
            for i, (w, b) in enumerate(zip(weights, biases)):
                layer = nn.Linear(w.shape[1], w.shape[0])
                with torch.no_grad():
                    layer.weight.copy_(torch.from_numpy(w))
                    layer.bias.copy_(torch.from_numpy(b))
                modules.append(layer)
                if i < len(weights) - 1:
                    modules.append(nn.ReLU())
            modules.append(nn.Tanh())
            return nn.Sequential(*modules)
        teacher = build(self.weights, self.biases).eval()
        student = build(self.pruned_weights, self.pruned_biases).train()
        linear_layers = [m for m in student if isinstance(m, nn.Linear)]
        masks = [torch.from_numpy(m) for m in self.masks]
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
        with torch.no_grad():
            labels = teacher(observations)
            initial_loss = float(torch.mean((student(observations) - labels) ** 2))
        history = []
        generator = torch.Generator().manual_seed(seed)
        for step in range(steps):
            indices = torch.randint(len(observations), (batch_size,), generator=generator)
            loss = torch.mean((student(observations[indices]) - labels[indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for layer, mask in zip(linear_layers, masks):
                layer.weight.grad.mul_(mask)
            optimizer.step()
            with torch.no_grad():
                for layer, mask in zip(linear_layers, masks):
                    layer.weight.mul_(mask)
            if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
                history.append({"step": step + 1, "batch_action_mse": float(loss.detach())})
        self.pruned_weights = [m.weight.detach().numpy().copy() for m in linear_layers]
        self.pruned_biases = [m.bias.detach().numpy().copy() for m in linear_layers]
        self.integer_layers = self._build_integer_layers(np.asarray(calibration_obs, dtype=np.float32))
        with torch.no_grad():
            final_loss = float(torch.mean((student(observations) - labels) ** 2))
        return {"method": "fixed-magnitude-mask action distillation from trained FP32 actor",
                "steps": steps, "batch_size": batch_size, "seed": seed,
                "initial_action_mse": initial_loss, "final_action_mse": final_loss,
                "history": history}

    def validation_report(self, obs):
        base = self.action(obs, "fp32")
        pruned = self.action(obs, "pruned_fp32")
        quant = self.action(obs, "int8")
        total = sum(w.size for w in self.weights)
        zeros = sum(np.count_nonzero(w == 0) for w in self.pruned_weights)
        first = self.integer_layers[0]
        raw = self._balanced_input(obs) / first.input_scale + first.input_zero
        return {
            "calibration_count": self.calibration_count,
            "evaluation_count": len(np.atleast_2d(obs)),
            "target_prune_ratio": self.prune_ratio,
            "actual_weight_sparsity": zeros / total,
            "weight_count": total,
            "zero_weight_count": int(zeros),
            "pruned_action_mae": float(np.mean(np.abs(base - pruned))),
            "pruned_action_max_abs_error": float(np.max(np.abs(base - pruned))),
            "int8_vs_pruned_action_mae": float(np.mean(np.abs(pruned - quant))),
            "int8_vs_fp32_action_mae": float(np.mean(np.abs(base - quant))),
            "int8_vs_fp32_action_max_abs_error": float(np.max(np.abs(base - quant))),
            "input_saturation_fraction": float(np.mean((raw < -128.5) | (raw > 127.5))),
            "input_balancing": "calibration_"+self.balance_mode+"_affine_folded_into_first_layer" if self.balance_inputs else "identity",
            "input_balancing_mode": self.balance_mode,
            "smoothquant_alpha": self.smoothquant_alpha if self.balance_mode == "smoothquant" else None,
            "input_balancing_center": self.input_center.tolist(),
            "input_balancing_channel_scale": self.input_channel_scale.tolist(),
            "input_balancing_selection_basis": "Configured exponent; laboratory calibration-MAE candidate scores are recorded in metrics.edge.quantization_selection. Closed-loop test values are not used for range calibration or exponent selection.",
            "smoothquant_formula": "s_j=max(max|cal_j-center_j|,eps)^alpha/max(max|W_pruned[:,j]|,eps)^(1-alpha); eps=1e-5",
            "constant_channel_scale_rule": "max centered absolute input <=1e-5 (mean_std: standard deviation <=1e-5) uses scale 1; channel is retained",
            "kernel": "INT8 weights/activations, INT32 accumulation, INT64 fixed-point requantization; float affine input prefix, input quantization and final tanh",
            "speed_scope": "Python/NumPy CPU reference, not a DSP/MCU deployment measurement",
        }

    def benchmark(self, obs, repeats=2000, warmup=100):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        results = {}
        for variant in ("fp32", "pruned_fp32", "int8"):
            for _ in range(warmup):
                self.action(obs, variant)
            elapsed = np.empty(repeats)
            for j in range(repeats):
                start = time.perf_counter_ns()
                self.action(obs, variant)
                elapsed[j] = (time.perf_counter_ns() - start) / 1e6
            results[variant] = {"median_ms": float(np.median(elapsed)),
                                "p95_ms": float(np.percentile(elapsed, 95)),
                                "min_ms": float(elapsed.min()),
                                "repeats": repeats, "batch_size": 1}
        return results

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for variant, weights, biases in (("fp32", self.weights, self.biases),
                                         ("pruned_fp32", self.pruned_weights, self.pruned_biases)):
            arrays = {f"weight_{i}": w for i, w in enumerate(weights)}
            arrays.update({f"bias_{i}": b for i, b in enumerate(biases)})
            np.savez_compressed(directory / f"actor_{variant}.npz", **arrays)
        arrays = {"input_center": self.input_center,
                  "input_channel_scale": self.input_channel_scale}
        metadata = {"format_version": 2, "prune_ratio": self.prune_ratio,
                    "input_balancing": "calibration_"+self.balance_mode+"_affine_folded_into_first_layer" if self.balance_inputs else "identity",
                    "input_balancing_mode": self.balance_mode,
                    "smoothquant_alpha": self.smoothquant_alpha if self.balance_mode == "smoothquant" else None,
                    "input_balancing_floor": self.input_balancing_floor,
                    "calibration_count": self.calibration_count, "layers": []}
        for i, layer in enumerate(self.integer_layers):
            arrays[f"weight_{i}"] = layer.weights
            arrays[f"bias_{i}"] = layer.bias
            metadata["layers"].append({key: getattr(layer, key) for key in
                ("input_scale", "input_zero", "weight_scale", "weight_zero", "output_scale",
                 "output_zero", "multiplier", "shift", "relu")})
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        np.savez_compressed(directory / "actor_int8.npz", **arrays)
        sizes = {name: (directory / name).stat().st_size for name in
                 ("actor_fp32.npz", "actor_pruned_fp32.npz", "actor_int8.npz")}
        return sizes

    @staticmethod
    def load_int8(path):
        """Return a standalone callable; no Torch or original float weights needed."""
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            layers = [IntegerLayer(data[f"weight_{i}"].copy(), data[f"bias_{i}"].copy(), **meta)
                      for i, meta in enumerate(metadata["layers"])]
            # Version 1 artifacts had no affine prefix. Never infer new
            # statistics when loading: their original integer layers require
            # the identity input transformation.
            input_dim = layers[0].weights.shape[1]
            center = data["input_center"].astype(np.float32).copy() if "input_center" in data else np.zeros(input_dim, np.float32)
            channel_scale = data["input_channel_scale"].astype(np.float32).copy() if "input_channel_scale" in data else np.ones(input_dim, np.float32)
            if center.shape != (input_dim,) or channel_scale.shape != (input_dim,):
                raise ValueError("Invalid INT8 affine input prefix shape")
            if not np.isfinite(center).all() or not np.isfinite(channel_scale).all() or np.any(channel_scale <= 0):
                raise ValueError("Invalid INT8 affine input prefix values")
        def action(obs):
            first = layers[0]
            raw = np.asarray(obs, dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError("Nonfinite observation")
            balanced = (raw - center) / channel_scale
            q = _quantize(balanced, first.input_scale, first.input_zero)
            for layer in layers:
                q = layer.forward(q)
            last = layers[-1]
            return np.tanh(last.output_scale * (q.astype(np.float32) - last.output_zero))
        return action

"""A fixed, nominally tuned conventional position PID baseline.

All gain units are explicit: Kp V/mm, Ki V/(mm*s), Kd V/(mm/s).
The previous cruise voltage is an initial integrator/bias operating point, not
an online load observer. Feedback uses only position and velocity error.
"""
from __future__ import annotations
import numpy as np


class Controller:
    def __init__(self, initial_voltage, dt=.001, kp=1., ki=4., kd=.01,
                 derivative_alpha=.85):
        self.initial_voltage = float(initial_voltage)
        self.dt = float(dt)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.derivative_alpha = float(derivative_alpha)
        self.integral_mm_s = 0.
        self.filtered_derivative_mm_s = 0.

    def update(self, position_error_mm, velocity_error_mms):
        error = float(position_error_mm)
        velocity_error = float(velocity_error_mms)
        if not np.isfinite([error, velocity_error]).all():
            raise ValueError('PID feedback must be finite')
        a = self.derivative_alpha
        self.filtered_derivative_mm_s = (a*self.filtered_derivative_mm_s +
                                         (1-a)*velocity_error)
        trial_integral = self.integral_mm_s + error*self.dt
        proposed = (self.initial_voltage + self.kp*error + self.ki*trial_integral +
                    self.kd*self.filtered_derivative_mm_s)
        # Conditional integration: do not accumulate further into saturation.
        # Retain integral updates that drive an already saturated output back.
        if abs(proposed) <= 10. or np.sign(error) != np.sign(proposed):
            self.integral_mm_s = trial_integral
        voltage = (self.initial_voltage + self.kp*error + self.ki*self.integral_mm_s +
                   self.kd*self.filtered_derivative_mm_s)
        return float(np.clip(voltage, -10., 10.))

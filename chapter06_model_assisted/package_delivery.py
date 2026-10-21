"""生成含源码、实验记录与图片的可核验ZIP（不会把ZIP嵌入自身）。"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    excluded = {"__pycache__", ".git", ".venv", "deliverables", ".mplconfig"}
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and not excluded.intersection(p.relative_to(ROOT).parts)
                   and p.name != "manifest.json" and p.suffix not in (".pyc", ".log"))
    manifest = {"format": 1, "files": {
        p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    delivery = ROOT / "deliverables"
    delivery.mkdir(exist_ok=True)
    target = delivery / "chapter06_model_assisted.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for p in files + [manifest_path]:
            archive.write(p, Path(ROOT.name) / p.relative_to(ROOT))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP校验失败")
    print(f"{target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

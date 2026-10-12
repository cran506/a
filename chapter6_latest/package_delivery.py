"""为当前第六章源码、单张结果图和原始数据生成完整交付 ZIP。"""
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
EXCLUDE_DIRS = {"deliverables", "__pycache__", "outputs_unknown"}


def main():
    files = sorted(path for path in ROOT.rglob("*") if path.is_file()
                   and not EXCLUDE_DIRS.intersection(path.relative_to(ROOT).parts)
                   and path.name != "manifest.json"
                   and path.suffix not in {".pyc", ".pyo"})
    required = {"README.md", "model_assisted_lab.py", "model_assisted_controller.py",
                "outputs/model_assisted_comparison.png", "outputs/summary.json",
                "outputs/pid.csv", "outputs/nominal.csv", "outputs/full.csv",
                "outputs/validation.json"}
    available = {path.relative_to(ROOT).as_posix() for path in files}
    if not required <= available:
        raise RuntimeError(f"交付文件尚未齐全：{sorted(required-available)}")
    manifest = {path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files}
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    directory = ROOT / "deliverables"
    directory.mkdir(exist_ok=True)
    target = directory / "第六章_模型辅助控制交付.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files+[manifest_path]:
            archive.write(path, (Path(ROOT.name)/path.relative_to(ROOT)).as_posix())
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"压缩包校验失败：{bad}")
    print(f"ZIP：{target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

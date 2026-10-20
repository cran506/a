"""Package source, trained artifacts and measured data with SHA-256 checksums."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile


def package(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    required=['outputs/chapter8_comparison.png','outputs/第八章实验报告.docx',
        'outputs/evaluation.csv','outputs/training.csv','models/sac_model.zip',
        'models/actor_int8.npz','outputs/metrics.json','outputs/validation.json']
    for name in required:
        if not (root/name).is_file():raise FileNotFoundError(root/name)
    directory=root/'deliverables';directory.mkdir(exist_ok=True)
    def included(path):
        relative=path.relative_to(root)
        return path.is_file() and not any(part in ['deliverables','__pycache__','.venv'] for part in relative.parts) and path.suffix not in ['.pyc','.so'] and path.name!='manifest.json'
    files=sorted(p for p in root.rglob('*') if included(p))
    manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    manifest_path=root/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    target=directory/'第八章_SAC安全控制完整交付.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files+[manifest_path]:archive.write(p,arcname=str(Path(root.name)/p.relative_to(root)))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise AssertionError('ZIP integrity failed')
    print(f'{target} ({target.stat().st_size} bytes, {len(files)+1} files)')
    return target

if __name__=='__main__':package()

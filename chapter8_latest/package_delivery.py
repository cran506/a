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
    validation=json.loads((root/'outputs/validation.json').read_text(encoding='utf-8'))
    if validation.get('checks',{}).get('all_checks_passed') is not True:
        raise RuntimeError('Current independent verification has not passed')
    for category in ('source_sha256','artifact_sha256','data_sha256'):
        entries=validation.get(category)
        if not isinstance(entries,dict) or not entries:
            raise RuntimeError('Verification is missing bound evidence: '+category)
        for name,expected in entries.items():
            path=root/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Verification is stale; re-run verify_experiment.py: '+name)
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
        for name,metadata in manifest.items():
            if hashlib.sha256(archive.read(str(Path(root.name)/name))).hexdigest()!=metadata['sha256']:
                raise AssertionError('ZIP content hash mismatch: '+name)
    print(f'{target} ({target.stat().st_size} bytes, {len(files)+1} files)')
    return target

if __name__=='__main__':package()

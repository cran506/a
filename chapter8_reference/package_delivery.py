"""将已验证代码、训练策略、原始日志与图打包；不包含临时模型。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile


def package(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    validation=json.loads((root/'outputs/validation.json').read_text())
    if not validation['checks']['all_checks_passed']:raise RuntimeError('验证尚未通过')
    for category in ('source_sha256','artifact_sha256','data_sha256'):
        for name,expected in validation[category].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('验证后文件已变动，请重新验证：'+name)
    for name in ('outputs/chapter8_comparison.png','outputs/chapter8_comparison.pdf'):
        if not (root/name).is_file():raise FileNotFoundError(name)
    def include(path):
        relative=path.relative_to(root)
        return (path.is_file() and not any(x in relative.parts for x in ('deliverables','__pycache__','checkpoints','trials'))
                and path.suffix not in ('.pyc','.so') and path.name not in ('manifest.json','sac_checkpoint.zip','progress.json'))
    files=sorted(p for p in root.rglob('*') if include(p))
    manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    target=root/'deliverables/第八章_五维参考接口修订版.zip';target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in files+[root/'manifest.json']:archive.write(p,str(Path(root.name)/p.relative_to(root)))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise AssertionError('ZIP 校验失败')
    print(target,flush=True);return target


if __name__=='__main__':package()

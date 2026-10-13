"""Setup-only downloads; no model imports or GPU use."""
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
import yaml

root=Path(__file__).resolve().parent
versions=json.loads((root/'versions.json').read_text())
cache=root/'cache/huggingface/hub'
snapshot_download(versions['model_repository'],revision=versions['model_revision'],local_dir=root/'checkpoints',cache_dir=cache)
cfg=yaml.safe_load((root/'checkpoints/config.yaml').read_text())
lockpath=root/'runtime_lock.json'
lock=json.loads(lockpath.read_text()) if lockpath.exists() else {}
repos=['facebook/w2v-bert-2.0','amphion/MaskGCT','funasr/campplus',cfg['vocoder']['name']]
for repo in repos:
    if repo not in lock:lock[repo]=HfApi().model_info(repo).sha
lockpath.write_text(json.dumps(lock,indent=2)+'\n')
for repo,rev in lock.items():
    if repo=='amphion/MaskGCT':
        hf_hub_download(repo,filename='semantic_codec/model.safetensors',revision=rev,cache_dir=cache)
    else:snapshot_download(repo,revision=rev,cache_dir=cache)
    # Upstream loads these repos without revision; pin main resolution in this isolated cache.
    path=cache/('models--'+repo.replace('/','--'))/'refs/main'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(rev)
print('Downloaded fixed IndexTTS-2 weights and locked auxiliary models')

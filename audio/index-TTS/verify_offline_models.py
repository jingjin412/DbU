"""Check IndexTTS's offline dependencies using CPU file/config access only."""
import json
import os
from pathlib import Path
from huggingface_hub import try_to_load_from_cache, constants
import yaml

ROOT=Path(__file__).resolve().parent


def main():
    expected=ROOT/'cache/huggingface/hub'
    alias=ROOT/'checkpoints/hf_cache'
    if not alias.is_symlink() or alias.resolve()!=expected.resolve():
        raise ValueError('IndexTTS hard-coded cache alias missing; run prepare_cache_layout.py')
    if Path(constants.HF_HUB_CACHE).resolve()!=expected.resolve():
        raise ValueError('Wrong HF_HUB_CACHE: '+constants.HF_HUB_CACHE)
    for key in ('TRANSFORMERS_CACHE','PYTORCH_TRANSFORMERS_CACHE','PYTORCH_PRETRAINED_BERT_CACHE'):
        if os.environ.get(key) and Path(os.environ[key]).resolve()!=expected.resolve():
            raise ValueError('Conflicting '+key+'; use run.sbatch')
    cfg=yaml.safe_load((ROOT/'checkpoints/config.yaml').read_text())
    lock=json.loads((ROOT/'runtime_lock.json').read_text())
    required={
        'facebook/w2v-bert-2.0':['preprocessor_config.json','config.json','model.safetensors'],
        'amphion/MaskGCT':['semantic_codec/model.safetensors'],
        'funasr/campplus':['campplus_cn_common.bin'],
        cfg['vocoder']['name']:['config.json','bigvgan_generator.pt'],
    }
    count=0
    for repo,files in required.items():
        if repo not in lock:raise ValueError('Missing runtime lock entry: '+repo+'; rerun setup.sbatch')
        ref=expected/('models--'+repo.replace('/','--'))/'refs/main'
        if not ref.is_file() or ref.read_text().strip()!=lock[repo]:
            raise ValueError('Offline main revision not pinned: '+repo)
        for filename in files:
            path=try_to_load_from_cache(repo,filename,revision='main',cache_dir=expected)
            if not isinstance(path,str) or not Path(path).is_file() or Path(path).stat().st_size==0:
                raise FileNotFoundError('Missing offline model file: '+repo+'/'+filename+'; rerun setup.sbatch')
            count+=1
    print(f'Offline cache verified: {count} files across {len(required)} auxiliary models; no GPU used')


if __name__=='__main__':main()

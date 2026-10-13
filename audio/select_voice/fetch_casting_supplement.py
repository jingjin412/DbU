#!/usr/bin/env python3
"""Download a small, pinned AISHELL-3 casting supplement. CPU only; no models."""
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download
import soundfile as sf
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO='shenyunhang/AISHELL-3'
REVISION='e1af0fa23933585f1844b3c07b71006f86589bab'
TARGETS=['SSB1100','SSB1630','SSB1365','SSB0966','SSB1831','SSB1328',
         'SSB1110','SSB1126','SSB1320','SSB1050']
DEST=ROOT/'sources/AISHELL-3-supplement'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    api=HfApi(token=False)
    # Confirm that the mirror's IDs and age/gender metadata describe the same corpus.
    meta=Path(hf_hub_download(REPO,'spk-info.txt',repo_type='dataset',revision=REVISION,token=False))
    def parse(p):
        return {x[0]:x[1:4] for line in p.read_text().splitlines() if not line.startswith('#') and len(x:=line.split())>=4}
    assert parse(meta)==parse(ROOT/'sources/AISHELL-3/spk-info.txt'),'AISHELL speaker metadata mismatch'
    shutil.copyfile(meta,DEST/'spk-info.txt')
    content=Path(hf_hub_download(REPO,'test/content.txt',repo_type='dataset',revision=REVISION,token=False))
    (DEST/'test').mkdir(exist_ok=True);shutil.copyfile(content,DEST/'test/content.txt')
    log_path=ROOT/'sources/casting_supplement_manifest.json'
    records=[]
    for speaker in TARGETS:
        entries=list(api.list_repo_tree(REPO,path_in_repo='test/wav/'+speaker,repo_type='dataset',revision=REVISION))
        # PCM WAV byte size approximates duration; avoid fetching many one-word clips.
        entries=sorted(entries,key=lambda e:abs(e.size-576044))
        picked=None
        for entry in entries[:12]:
            if not entry.path.endswith('.wav'):continue
            local=DEST/entry.path
            local.parent.mkdir(parents=True,exist_ok=True)
            if not local.is_file():
                cached=hf_hub_download(REPO,entry.path,repo_type='dataset',revision=REVISION,token=False)
                shutil.copyfile(cached,local)
            actual=digest(local);lfs=getattr(entry,'lfs',None)
            expected=getattr(lfs,'sha256',None) if lfs else None
            if isinstance(lfs,dict):expected=lfs.get('sha256') or lfs.get('oid')
            if expected and actual!=expected:raise ValueError('HF LFS hash mismatch: '+entry.path)
            values,rate=sf.read(local,dtype='float32')
            duration=len(values)/rate;rms=float(np.sqrt(np.mean(values**2)))
            clipped=float(np.mean(np.abs(values)>=.999))
            record={'speaker_id':speaker,'repo':REPO,'revision':REVISION,'repo_path':entry.path,
                    'local_path':str(local.relative_to(ROOT)),'sha256':actual,'lfs_sha256':expected,
                    'duration_seconds':duration,'rms':rms,'clipped_fraction':clipped,
                    'usable':3<=duration<=10 and rms>=.01 and clipped<=.001}
            records.append(record)
            if record['usable']:picked=record;break
        if not picked:raise ValueError('No usable reference for '+speaker)
        payload={'repo':REPO,'revision':REVISION,'metadata_matches_primary_snapshot':True,
                 'target_speakers':TARGETS,'files':records,'full_dataset_downloaded':False}
        log_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'speaker':speaker,'duration':round(picked['duration_seconds'],3),'sha256':picked['sha256']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

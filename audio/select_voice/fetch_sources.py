#!/usr/bin/env python3
"""CPU-only pinned HF fetch; verifies cached AISHELL audio and downloads Bazinga WAVs."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
VOICE = ROOT.parents[1]
OLD = Path(os.environ.get('VOICE_AISHELL_CACHE', str(ROOT/'sources/AISHELL-3')))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    token_path = Path.home()/'.cache/huggingface/token'
    token = os.environ.get('HF_TOKEN') or (token_path.read_text().strip() if token_path.exists() else '')
    def request(url):
        return urllib.request.Request(url, headers={'Authorization':'Bearer '+token} if token else {})
    repo = 'AISHELL/AISHELL-3'
    rev = json.loads((ROOT/'sources/AISHELL--AISHELL-3.json').read_text())['sha']
    inventory = ROOT/'sources/aishell_file_inventory.json'
    if inventory.exists(): files=json.loads(inventory.read_text())
    else:
        url=f'https://huggingface.co/api/datasets/{repo}/tree/{rev}?recursive=true&limit=1000'
        files=[]
        while url:
            with urllib.request.urlopen(request(url),timeout=60) as response:
                files.extend(r for r in json.load(response) if r['type']=='file')
                link=response.headers.get('Link','')
                url=next((s.split('>')[0].strip().lstrip('<') for s in link.split(',') if 'rel="next"' in s),None)
        inventory.write_text(json.dumps(files,indent=2))
    jobs=[]
    for f in files:
        jobs.append({'dataset':repo,'revision':rev,'source_path':f['path'],'bytes':f['size'],
                     'expected_sha256':f.get('lfs',{}).get('oid'),'destination':str(ROOT/'sources/AISHELL-3'/f['path']),
                     'reuse':str(OLD/f['path'])})
    baz=json.loads((VOICE/'datagen/raw/source_manifest.json').read_text())
    episodes=json.loads((VOICE/'data/bazinga/metadata.json').read_text())
    for episode,m in episodes.items():
        path=f'data/{m["series"]}/{episode}.en.wav'
        jobs.append({'dataset':'bazinga/bazinga','revision':baz['revision'],'source_path':path,
                     'destination':str(ROOT/'sources/bazinga'/Path(path).name)})
    def fetch(job):
        dest=Path(job['destination']);dest.parent.mkdir(parents=True,exist_ok=True)
        cached=Path(job['reuse']) if 'reuse' in job else None
        expected=job.get('expected_sha256')
        if dest.exists() and (not job.get('bytes') or dest.stat().st_size==job['bytes']):
            checksum=sha(dest)
            if not expected or checksum==expected: return {**job,'sha256':checksum,'action':'existing_verified'}
        if cached and cached.exists() and cached.stat().st_size==job['bytes']:
            checksum=sha(cached)
            if expected and checksum==expected:
                # Symlink reuse is explicit and final selected clips are independent copies.
                if not dest.exists():dest.symlink_to(cached.resolve())
                return {**job,'sha256':checksum,'action':'verified_local_reuse'}
        url=f'https://huggingface.co/datasets/{job["dataset"]}/resolve/{job["revision"]}/{job["source_path"]}'
        part=dest.with_suffix(dest.suffix+'.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request(url),timeout=120) as response, part.open('wb') as output:
                    shutil.copyfileobj(response,output,1024*1024)
                checksum=sha(part)
                if expected and checksum!=expected:raise ValueError('Checksum mismatch '+str(dest))
                if job.get('bytes') and part.stat().st_size!=job['bytes']:raise ValueError('Size mismatch')
                part.replace(dest)
                return {**job,'sha256':checksum,'action':'downloaded'}
            except Exception:
                if attempt==2:raise
                time.sleep(attempt+1)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(fetch,jobs),1):
            records.append(r)
            if i%500==0 or i>len(files):print(f'{i}/{len(jobs)} {r["source_path"]} {r["action"]}',flush=True)
    (ROOT/'sources/download_manifest.json').write_text(json.dumps(records,indent=2))
    print('Completed',len(records),flush=True)


if __name__=='__main__':main()

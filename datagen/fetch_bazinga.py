#!/usr/bin/env python3
"""Download official transcripts only; authenticate using existing HF login."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SHOWS = ('GameOfThrones', 'SixFeetUnder', 'TheOffice')
REPO = 'https://huggingface.co/datasets/bazinga/bazinga'


def main():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    token_path = Path(os.environ.get('HF_TOKEN_PATH', str(Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))/'token')))
    if not token and token_path.is_file():
        token = token_path.read_text().strip()
    headers = {'Authorization': f'Bearer {token}'} if token else {}

    def get(url):
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                    return response.read()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    manifest_path = ROOT/'raw/source_manifest.json'
    if manifest_path.exists():
        revision = json.loads(manifest_path.read_text())['revision']
    else:
        revision = json.loads(get('https://huggingface.co/api/datasets/bazinga/bazinga'))['sha']

    def fetch(path):
        url = f'{REPO}/resolve/{revision}/{path}'
        data = get(url)
        dest = ROOT/'raw'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {'path': path, 'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    records = [fetch('README.md'), fetch('bazinga.py')]
    paths = []
    for show in SHOWS:
        records.extend(fetch(f'data/{show}/{name}.txt') for name in ('episodes', 'characters', 'credits'))
        episodes = (ROOT/f'raw/data/{show}/episodes.txt').read_text().splitlines()
        paths.extend(f'data/{show}/{line.split(",", 1)[0]}.txt' for line in episodes if line.strip())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i, record in enumerate(pool.map(fetch, paths), 1):
            records.append(record)
            if i % 20 == 0 or i == len(paths):
                print(f'Downloaded {i}/{len(paths)} transcripts', flush=True)
    manifest_path.write_text(json.dumps({'repository': REPO, 'revision': revision, 'files': records}, indent=2)+'\n')
    print(f'Pinned revision: {revision}')


if __name__ == '__main__':
    main()

"""Connect IndexTTS's hard-coded cache path to our pinned offline cache. CPU only."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    target=ROOT/'cache/huggingface/hub'
    target.mkdir(parents=True,exist_ok=True)
    link=ROOT/'checkpoints/hf_cache'
    link.parent.mkdir(parents=True,exist_ok=True)
    if link.is_symlink() and link.resolve()==target.resolve():
        print('IndexTTS cache alias verified: '+str(link));return
    if link.exists() or link.is_symlink():
        backup=link.with_name('hf_cache.previous')
        index=1
        while backup.exists() or backup.is_symlink():
            index+=1;backup=link.with_name(f'hf_cache.previous_{index}')
        link.rename(backup)
        print('Preserved previous cache path: '+str(backup))
    link.symlink_to(target,target_is_directory=True)
    print('IndexTTS cache alias prepared: '+str(link)+' -> '+str(target))


if __name__=='__main__':main()

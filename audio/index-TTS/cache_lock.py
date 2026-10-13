"""Coordinate cache publication across Slurm jobs; never delete another owner's lock."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time


@contextmanager
def cache_entry_lock(audio, lock, timeout=1800, poll_seconds=1):
    audio, lock = Path(audio), Path(lock)
    started = time.monotonic()
    owns_lock = False
    try:
        while True:
            if audio.is_file():
                yield False
                return
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if time.monotonic()-started >= timeout:
                    raise TimeoutError(
                        f'Cache lock wait exceeded {timeout}s: {lock}; inspect owner/job, '
                        'do not remove a live lock')
                if time.monotonic()-started < poll_seconds:
                    print(f'Waiting for shared cache entry: {lock}', flush=True)
                time.sleep(poll_seconds)
                continue
            owns_lock = True
            with os.fdopen(fd, 'w') as stream:
                json.dump({'job_id': os.environ.get('SLURM_JOB_ID'),
                           'pid': os.getpid(), 'host': socket.gethostname(),
                           'created_at': time.time()}, stream)
            # Publication can race our acquisition after the prior owner releases.
            yield not audio.is_file()
            return
    finally:
        if owns_lock:
            lock.unlink(missing_ok=True)

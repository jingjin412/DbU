"""CPU-only concurrency tests; execute with Slurm rather than on the login node."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from cache_lock import cache_entry_lock


class CacheLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.audio = Path(self.tmp.name)/'clip.wav'
        self.lock = self.audio.with_suffix('.lock')

    def test_concurrent_waiter_reuses_atomic_publication(self):
        entered = threading.Event()
        results, errors = [], []
        def waiter():
            entered.set()
            try:
                with cache_entry_lock(self.audio, self.lock, timeout=2, poll_seconds=.01) as owns:
                    results.append(owns)
                    self.assertEqual(self.audio.read_bytes(), b'complete')
            except BaseException as error:
                errors.append(error)
        with cache_entry_lock(self.audio, self.lock) as owns:
            self.assertTrue(owns)
            self.assertEqual(json.loads(self.lock.read_text())['pid'], os.getpid())
            thread = threading.Thread(target=waiter)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.audio.with_suffix('.part').write_bytes(b'complete')
            self.audio.with_suffix('.part').replace(self.audio)
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(results, [False])
        self.assertFalse(self.lock.exists())

    def test_failed_owner_releases_lock_for_retry(self):
        with self.assertRaises(KeyboardInterrupt):
            with cache_entry_lock(self.audio, self.lock):
                raise KeyboardInterrupt
        self.assertFalse(self.lock.exists())
        with cache_entry_lock(self.audio, self.lock) as owns:
            self.assertTrue(owns)

    def test_timeout_preserves_foreign_lock(self):
        self.lock.write_text('foreign owner')
        with self.assertRaises(TimeoutError):
            with cache_entry_lock(self.audio, self.lock, timeout=.02, poll_seconds=.01):
                self.fail('must not enter')
        self.assertEqual(self.lock.read_text(), 'foreign owner')

    def test_existing_audio_bypasses_lock(self):
        self.audio.write_bytes(b'complete')
        self.lock.write_text('foreign owner')
        with cache_entry_lock(self.audio, self.lock) as owns:
            self.assertFalse(owns)
        self.assertEqual(self.lock.read_text(), 'foreign owner')


if __name__ == '__main__':
    unittest.main()

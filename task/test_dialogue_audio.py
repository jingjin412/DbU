import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import wave

import numpy as np
import soundfile as sf
from dialogue_audio import LAYOUT, keep_ranges, mapped_frame
from model_io import Client
import hashlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrimmedDialogueTest(unittest.TestCase):
    def test_long_gaps_trimmed_short_gaps_kept(self):
        turns = [{'start_time': a, 'end_time': b} for a, b in [(1, 2), (3, 4), (20, 21)]]
        ranges = keep_ranges(turns, 100, 2120, 3000)
        self.assertEqual(ranges, [[80, 420], [1980, 2120]])
        self.assertEqual(mapped_frame(2100, ranges), 460)
        self.assertEqual(mapped_frame(2000, ranges), 360)

    def test_padding_never_crosses_causal_cutoff(self):
        turns = [{'start_time': 1., 'end_time': 2.}]
        self.assertEqual(keep_ranges(turns, 100, 205, 1000), [[80, 205]])

    def test_transport_encodes_only_current_prefix_not_future_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'dialogue.wav'
            original = np.concatenate([np.full(16000, .2), np.full(16000, .8)]).astype('float32')
            sf.write(path, original, 16000, subtype='PCM_16')
            entry = {'path': str(path), 'audio_sha256': sha(path), 'audio_layout': LAYOUT,
                     'sample_rate': 16000, 'prefix_frames': 16000, 'covered_utterance_ids': ['u0'],
                     'timeline': [{'index': 0, 'speaker': 'A', 'start_frame': 0, 'end_frame': 16000}],
                     'source': 'real_audio', 'fallback': False}
            manifest = root / 'audio_manifest.json'; manifest.write_text(json.dumps({'u0': entry}))
            sample = {'task': 'next_speaker_ranking', 'history': [{'index': 0, 'speaker': 'A', 'utterance': 'hi', 'utterance_id': 'u0'}],
                      'candidates': ['A', 'B'], 'context_start': 0}
            args = SimpleNamespace(audio_manifest=manifest, condition='real_audio', audio_layout='dialogue_prefix',
                                   audio_scope='history', model_kind='qwen3-instruct', max_audio_clips=160, output_format='unconstrained')
            client = Client(args)
            content = client.messages(sample)[0]['content']
            encoded = next(item['audio_url']['url'].split(',', 1)[1] for item in content if item['type']=='audio_url')
            values, rate = sf.read(io.BytesIO(base64.b64decode(encoded)))
            self.assertEqual(len(values), 16000)
            self.assertTrue(np.all(values < .3))
            self.assertEqual(client.audio_usage(sample)[0]['encoded_frame_end'], 16000)
            with self.assertRaises(ValueError):
                Client(SimpleNamespace(**dict(vars(args), audio_layout='utterance')))


if __name__ == '__main__':
    unittest.main()

import hashlib
import base64
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from model_io import Client
from data import samples
from run import run_dataset


class AudioContracts(unittest.TestCase):
    def test_phi4_per_turn_audio_and_duration_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history, manifest, expected = [], {}, []
            for index in range(2):
                values = np.full(160, (index+1)/10, dtype='float32')
                path = root/f'{index}.wav'
                sf.write(path, values, 16000, subtype='FLOAT')
                manifest[str(index)] = str(path)
                history.append({'index': index, 'speaker': f'角色{index}',
                                'utterance': 'VISIBLE', 'utterance_id': str(index)})
                expected.append(values)
            manifest['FUTURE'] = manifest['0']
            path = root/'manifest.json'; path.write_text(json.dumps(manifest))
            args = SimpleNamespace(audio_manifest=path, model_kind='phi-4-multimodal-instruct',
                                   condition='neutral_audio', audio_scope='history', max_audio_clips=32,
                                   output_format='plain')
            sample = {'task': 'next_speaker_ranking', 'history': history,
                      'candidates': ['角色0', '角色1', '角色2'], 'naming': 'original'}
            content = Client(args).messages(sample)[0]['content']
            audios = [part for part in content if part['type'] == 'audio_url']
            self.assertEqual(len(audios), 2)
            self.assertEqual(content[-1]['type'], 'text')
            self.assertIn('Who will speak NEXT?', content[-1]['text'])
            for part, wave in zip(audios, expected):
                actual, sr = sf.read(io.BytesIO(base64.b64decode(part['audio_url']['url'].split(',', 1)[1])), dtype='float32')
                self.assertEqual(sr, 16000)
                np.testing.assert_array_equal(actual, wave)
            self.assertNotIn('FUTURE', str(content))
            self.assertNotIn('JSON', str(content))
            sf.write(root/'0.wav', np.zeros(41*16000, dtype='float32'), 16000)
            with self.assertRaisesRegex(ValueError, 'overlong audio'):
                Client(args).clips(sample)

    def test_flamingo_single_audio_preserves_history_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history, manifest, expected = [], {}, []
            for index in range(6):
                values = np.full((160, 1), (index+1)/20, dtype='float32')
                wav = root/f'{index}.wav'
                sf.write(wav, values, 16000, subtype='FLOAT')
                manifest[str(index)] = str(wav)
                history.append({'index': index, 'speaker': f'角色{index}',
                                'utterance': f'Visible {index}', 'utterance_id': str(index)})
                expected.append(values)
            manifest['SECRET_FUTURE'] = manifest['0']
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            args = SimpleNamespace(audio_manifest=path, model_kind='audio-flamingo-3',
                                   condition='neutral_audio', audio_scope='history', max_audio_clips=160)
            sample = {'task': 'next_speaker_ranking', 'history': history,
                      'candidates': [turn['speaker'] for turn in history], 'naming': 'original'}
            content = Client(args).messages(sample)[0]['content']
            inputs = [part for part in content if part['type'] == 'audio_url']
            self.assertEqual(len(inputs), 1)
            encoded = inputs[0]['audio_url']['url'].split(',', 1)[1]
            values, sr = sf.read(io.BytesIO(base64.b64decode(encoded)), dtype='float32', always_2d=True)
            self.assertEqual(sr, 16000)
            np.testing.assert_array_equal(values, np.concatenate(expected))
            labels = '\n'.join(part['text'] for part in content if part['type'] == 'text')
            for index in range(6): self.assertIn(f'utterance {index+1}, speaker 角色{index}', labels)
            self.assertNotIn('SECRET_FUTURE', str(content))
            # Each individual clip fits, but the full request would be truncated by AF3.
            for index in range(6): sf.write(root/f'{index}.wav', np.zeros(101*16000), 16000)
            with self.assertRaisesRegex(ValueError, 'exceeds 600 seconds'):
                Client(args).clips(sample)
            sf.write(root/'0.wav', np.zeros(100), 8000)
            with self.assertRaisesRegex(ValueError, 'samplerate/channels'):
                Client(args).clips(sample)

    def test_voxtral_groups_all_visible_audio_without_future_or_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history, manifest, expected = [], {}, []
            for index in range(12):
                values = np.full((160, 1), (index+1)/20, dtype='float32')
                wav = root/f'{index}.wav'
                sf.write(wav, values, 16000, subtype='FLOAT')
                manifest[str(index)] = str(wav)
                history.append({'index': index, 'speaker': '左母' if index%2 else '左父',
                                'utterance': f'Visible {index}', 'utterance_id': str(index)})
                expected.append(values)
            future = root/'future.wav'
            sf.write(future, np.ones((160,1), dtype='float32'), 16000, subtype='FLOAT')
            manifest['SECRET_FUTURE'] = str(future)
            manifest_path = root/'audio.json'
            manifest_path.write_text(json.dumps(manifest))
            args = SimpleNamespace(audio_manifest=manifest_path, model_kind='voxtral-small-24b',
                                   condition='neutral_audio', audio_scope='history', max_audio_clips=160)
            client = Client(args)
            sample = {'task': 'addressee_recognition', 'history': history,
                      'candidates': ['左母','左父'], 'naming': 'original'}
            content = client.messages(sample)[0]['content']
            inputs = [part for part in content if part['type'] == 'input_audio']
            self.assertLessEqual(len(inputs), 5)
            decoded = [sf.read(io.BytesIO(base64.b64decode(part['input_audio']['data'])),
                               dtype='float32', always_2d=True)[0] for part in inputs]
            np.testing.assert_array_equal(np.concatenate(decoded), np.concatenate(expected))
            self.assertNotIn('SECRET_FUTURE', str(content))
            labels = '\n'.join(part['text'] for part in content if part['type']=='text')
            for index in range(12): self.assertIn(f'utterance {index+1}, speaker', labels)
            short = {**sample, 'history': history[:5]}
            short_content = client.messages(short)[0]['content']
            short_inputs = [part for part in short_content if part['type']=='input_audio']
            self.assertEqual(len(short_inputs),5)
            for index, part in enumerate(short_inputs):
                values, _ = sf.read(io.BytesIO(base64.b64decode(part['input_audio']['data'])), dtype='float32', always_2d=True)
                np.testing.assert_array_equal(values, expected[index])
            self.assertIn('segment starting at frame 0', str(short_content))

    def test_audio_identity_fallback_cutoff_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wav = root/'neutral.wav'
            sf.write(wav, np.zeros(1600), 16000)
            entry = {'path': str(wav), 'source': 'neutral_audio', 'fallback': True,
                     'fallback_reason': 'predicted_audio_not_generated', 'emotion_model': 'qwen3omni',
                     'audio_sha256': hashlib.sha256(wav.read_bytes()).hexdigest()}
            manifest = root/'audio.json'
            manifest.write_text(json.dumps({'bazinga:d:u00001': entry, 'future': entry}))
            data = root/'data.json'
            data.write_text(json.dumps({'d': [
                {'speaker': 'A', 'listener': [{'name': 'B'}, {'name': 'C'}], 'utterance': 'Hi'},
                {'speaker': 'B', 'listener': [{'name': 'A'}], 'utterance': 'SECRET_FUTURE'}]}))
            metadata = root/'metadata.json'
            metadata.write_text(json.dumps({'d': {'utterances': [{'utterance_id': 'bazinga:d:u00001'}, {'utterance_id': 'future'}]}}))
            args = SimpleNamespace(data=data, metadata=metadata, gold_emotions=None, dataset='bazinga',
                task='next_speaker_ranking', min_participants=3, naming='original', context_turns=0,
                max_dialogues=0, max_samples=0, model_kind='qwen3-instruct', model=None,
                condition='predicted_audio', audio_manifest=manifest, audio_scope='history',
                max_audio_clips=64, served_model='voice-model', endpoint='unused', concurrency=1,
                timeout=300, max_new_tokens=512, temperature=0, seed=42,
                output_dir=root/'out', run_id='test', resume=False, dry_run=False)
            args.backend = 'http'
            args.runtime_metadata = None
            args.max_model_len = 32768
            args.smoke_audio_coverage = False
            args.fail_fast_requests = False
            args.reuse_from_run = None
            captured = []
            def response(client, route, payload):
                captured.append(payload)
                return {'choices': [{'message': {'content': '{"ranking":["B","A","C"]}'}, 'finish_reason': 'stop'}]}
            with patch.object(Client, 'check_server', return_value=[]), patch.object(Client, 'request', response):
                run_dataset(args)
                args.resume = True
                run_dataset(args)
            self.assertEqual(len(captured), 1)
            content = captured[0]['messages'][0]['content']
            self.assertNotIn('SECRET_FUTURE', str(content))
            self.assertEqual(sum(c['type'] == 'audio_url' for c in content), 1)
            folder = args.output_dir/args.task/'bazinga'/'qwen3-instruct'/'original'/'test'
            row = json.loads((folder/'predictions.jsonl').read_text())
            self.assertTrue(row['audio_fallback'])
            self.assertEqual(row['audio_fallback_count'], 1)
            self.assertEqual(row['audio_usage'][0]['fallback_reason'], 'predicted_audio_not_generated')
            self.assertEqual(json.loads((folder/'metrics.json').read_text())['audio_neutral_fallback_utterances'], 1)
            sample = {'history': [{'index': 0, 'speaker': 'A', 'utterance_id': 'bazinga:d:u00001'}]}
            args.model_kind = 'gemma4-e4b'
            with self.assertRaisesRegex(ValueError, 'model differs'):
                Client(args).clips(sample)
            args.model_kind = 'qwen3-instruct'
            args.model_kind = 'voxtral-small-24b'
            manifest.write_text(json.dumps({'bazinga:d:u00001': {**entry, 'emotion_model': 'voxtral'},
                                           'future': {**entry, 'emotion_model': 'voxtral'}}))
            voxtral_sample = samples(args, 'next_speaker_ranking')[0]
            voxtral_content = Client(args).messages(voxtral_sample)[0]['content']
            self.assertEqual(sum(part['type'] == 'input_audio' for part in voxtral_content), 1)
            self.assertEqual(sum(part['type'] == 'audio_url' for part in voxtral_content), 0)
            self.assertNotIn('SECRET_FUTURE', str(voxtral_content))
            audio = next(part for part in voxtral_content if part['type'] == 'input_audio')
            self.assertEqual(audio['input_audio']['format'], 'wav')
            args.model_kind = 'qwen3-instruct'
            manifest.write_text(json.dumps({'bazinga:d:u00001': entry, 'future': entry}))
            sf.write(wav, np.ones(1600), 16000)
            with self.assertRaisesRegex(ValueError, 'Audio changed'):
                Client(args).clips(sample)


if __name__ == '__main__':
    unittest.main()

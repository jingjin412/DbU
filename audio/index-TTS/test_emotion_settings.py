import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_jobs
import prepare_gold_emotion
import synthesize
import synthesize_parallel
from emotion_settings import ORDER, LABELS, validate_jobs


class FixedStrengthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tts = self.root/'audio/index-TTS'
        self.select = self.root/'audio/select_voice'
        for path in (self.tts, self.select/'manifests', self.root/'data/mpdd'):
            path.mkdir(parents=True)
        self.versions = {'emotion_vector_order': ORDER}
        self.write(self.tts/'versions.json', self.versions)
        self.reference = self.select/'reference.wav'
        self.reference.write_bytes(b'checksum-only CPU fixture')
        self.data = {'1': [{'speaker': 'Alice', 'utterance': label,
                           'listener': [{'name': 'Bob'}, {'name': 'Carol'}]}
                          for label in LABELS]}
        self.write(self.root/'data/mpdd/dialogue.json', self.data)
        self.write(self.select/'manifests/mpdd_stories.json',
                   [{'story_id': 'story', 'dialogue_ids': ['1']}])
        self.write(self.select/'manifests/mpdd_voices.json', [{
            'story_id': 'story', 'bindings': [{'source_label': 'Alice', 'dialogue_ids': ['1']}],
            'character_id': 'alice', 'reference_wav': 'reference.wav',
            'reference_sha256': hashlib.sha256(self.reference.read_bytes()).hexdigest(),
            'review_status': 'human_review_accepted'}])

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def plan(self, predictions=None):
        output = self.tts/'neutral.json'
        argv = ['prepare_jobs.py', '--dataset', 'mpdd', '--output', str(output)]
        if predictions:
            argv += ['--predictions', str(predictions)]
        with patch.object(prepare_jobs, 'ROOT', self.tts), \
             patch.object(prepare_jobs, 'VOICE', self.root), \
             patch.object(prepare_jobs, 'SELECT', self.select), \
             patch('sys.argv', argv):
            prepare_jobs.main()
        return json.loads(output.read_text())

    def test_neutral_and_predicted_plans(self):
        neutral = self.plan()
        validate_jobs(neutral['jobs'])
        self.assertEqual({tuple(j['emotion_vector']) for j in neutral['jobs']},
                         {(0, 0, 0, 0, 0, 0, 0, .6)})
        predictions = self.root/'predictions.jsonl'
        predictions.write_text('\n'.join(json.dumps({
            'target_utterance_id': f'mpdd:1:u{i+1:05d}', 'status': 'ok',
            'prediction': label}) for i, label in enumerate(LABELS)))
        self.write(self.root/'config.json', {
            'task': 'emotion_recognition', 'dataset': 'mpdd', 'condition': 'text',
            'data_sha256': prepare_jobs.digest(self.root/'data/mpdd/dialogue.json')})
        predicted = self.plan(predictions)
        validate_jobs(predicted['jobs'])
        self.assertEqual([j['emotion'] for j in predicted['jobs']], LABELS)

    def test_gold_plan_uses_point_six_for_every_class(self):
        base = self.plan()
        base_path = self.tts/'base.json'
        self.write(base_path, base)
        gold = {'1': [{**turn, 'emotion': label}
                      for turn, label in zip(self.data['1'], LABELS)]}
        gold_path = self.root/'gold.json'
        self.write(gold_path, gold)
        plan = prepare_gold_emotion.prepare(base_path, self.root/'data/mpdd/dialogue.json',
                                           gold_path, self.tts/'versions.json')
        validate_jobs(plan['jobs'])
        self.assertEqual([j['emotion'] for j in plan['jobs']], LABELS)

    def test_both_runners_reject_old_strength_before_reference_or_gpu(self):
        plan = self.plan()
        plan['jobs'][0]['emotion_vector'][-1] = .5
        path = self.tts/'old.json'
        self.write(path, plan)
        for runner in (synthesize, synthesize_parallel):
            with self.subTest(runner=runner.__name__), patch('sys.argv', [
                    runner.__name__, '--jobs', str(path), '--output-dir', str(self.root/'out'),
                    '--validate-only']), self.assertRaisesRegex(ValueError, 'strength 0.6'):
                runner.main()
        self.assertFalse((self.root/'out').exists())

    def test_wrong_label_vector_and_alpha_are_rejected(self):
        job = self.plan()['jobs'][0]
        for field, value in [('emotion_vector', [.6, 0, 0, 0, 0, 0, 0, 0]),
                             ('emo_alpha', .6), ('emotion', 'unknown')]:
            invalid = copy.deepcopy(job)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_jobs([invalid])


if __name__ == '__main__':
    unittest.main()

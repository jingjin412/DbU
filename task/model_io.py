"""OpenAI-compatible text/audio transport for the installed vLLM models."""
import base64
import io
import json
import os
from pathlib import Path
import urllib.request
import urllib.error
from tasks import prompt, output_schema

VOXTRAL_AUDIO_LAYOUT = 'voice-voxtral-per-turn-up-to-5-else-grouped-history-v1'
FLAMINGO_AUDIO_LAYOUT = 'voice-flamingo-single-visible-history-timeline-v1'
PHI4_AUDIO_LAYOUT = 'voice-phi4mm-per-turn-visible-history-instruction-last-speech-lora-v2'


def audio_groups(clips, max_chunks=5):
    """Keep all clips in order, grouping only when the model's item cap is exceeded."""
    size = max(1, (len(clips)+max_chunks-1)//max_chunks)
    return [clips[start:start+size] for start in range(0, len(clips), size)]


class Client:
    def __init__(self, args):
        self.args = args
        self.audio = json.loads(args.audio_manifest.read_text()) if args.audio_manifest else {}
        self.audio_info = {}
        if any(isinstance(v, dict) and v.get('audio_layout') for v in self.audio.values()) and getattr(args, 'audio_layout', 'utterance') != 'dialogue_prefix':
            raise ValueError('Dialogue-prefix manifest requires --audio-layout dialogue_prefix')

    def request(self, route, payload=None):
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('VOICE_API_KEY'):
            headers['Authorization'] = 'Bearer '+os.environ['VOICE_API_KEY']
        req = urllib.request.Request(self.args.endpoint.rstrip('/')+route,
              data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.args.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode(errors="replace")[:2000]}') from exc

    def check_server(self):
        models = self.request('/models')['data']
        model = next((m for m in models if m['id'] == self.args.served_model), None)
        if model is None:
            raise ValueError('Requested served model not found')
        if self.args.model and model.get('root') and Path(model['root']).resolve() != Path(self.args.model).resolve():
            raise ValueError('Server checkpoint differs from requested model')
        return models

    def clips(self, sample):
        if self.args.condition == 'text':
            return []
        if getattr(self.args, 'audio_layout', 'utterance') == 'dialogue_prefix':
            from dialogue_audio import prefix_clips
            return prefix_clips(self, sample)
        history = sample['history'] if self.args.audio_scope == 'history' else sample['history'][-1:]
        clips = []
        import soundfile as sf
        for turn in history:
            value = self.audio.get(turn['utterance_id'])
            if not value:
                raise ValueError(f'Missing audio for {turn["utterance_id"]}')
            entry = value if isinstance(value, dict) else {'path': value}
            expected_model = {'gemma4-e4b': 'gemma4e4b', 'qwen3-instruct': 'qwen3omni',
                              'voxtral-small-24b': 'voxtral', 'audio-flamingo-3': 'audio-flamingo-3',
                              'phi-4-multimodal-instruct': 'phi-4-multimodal-instruct'}.get(self.args.model_kind)
            if entry.get('emotion_model') and entry['emotion_model'] != expected_model:
                raise ValueError('Audio emotion model differs from prediction model')
            path = Path(entry['path'])
            if not path.is_absolute():
                path = self.args.audio_manifest.parent/path
            if path not in self.audio_info:
                if entry.get('audio_sha256'):
                    import hashlib
                    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['audio_sha256']:
                        raise ValueError(f'Audio changed since manifest was frozen: {path}')
                self.audio_info[path] = sf.info(path)
            info = self.audio_info[path]
            limit = 30 if self.args.model_kind == 'gemma4-e4b' else (600 if self.args.model_kind == 'audio-flamingo-3' else 300)
            if self.args.model_kind == 'phi-4-multimodal-instruct':
                limit = 40  # Conservative per-clip smoke policy from the official model card.
            if info.frames <= 0 or (info.duration > limit and self.args.model_kind != 'gemma4-e4b'):
                raise ValueError(f'Empty/overlong audio (no truncation): {path}')
            step = int(limit*info.samplerate)
            for start in range(0, info.frames, step):
                clips.append((turn, path, start, min(start+step, info.frames)))
        if len(clips) > self.args.max_audio_clips:
            raise ValueError(f'Too many audio clips: {len(clips)}')
        if self.args.model_kind == 'audio-flamingo-3' and clips:
            infos = [self.audio_info[path] for _, path, _, _ in clips]
            if len({(info.samplerate, info.channels) for info in infos}) != 1:
                raise ValueError('Flamingo history must share samplerate/channels; no automatic resampling')
            if sum(end-start for _, _, start, end in clips) > 600*infos[0].samplerate:
                raise ValueError('Flamingo visible audio exceeds 600 seconds (no truncation)')
        return clips

    def audio_usage(self, sample):
        if self.args.condition == 'text':
            return []
        if getattr(self.args, 'audio_layout', 'utterance') == 'dialogue_prefix':
            from dialogue_audio import prefix_entry
            turn, entry = prefix_entry(self, sample)
            return [{'utterance_id': turn['utterance_id'], **entry, 'encoded_frame_start': 0, 'encoded_frame_end': entry['prefix_frames']}]
        history = sample['history'] if self.args.audio_scope == 'history' else sample['history'][-1:]
        result = []
        for turn in history:
            value = self.audio[turn['utterance_id']]
            entry = value if isinstance(value, dict) else {'path': value, 'source': self.args.condition, 'fallback': False}
            result.append({'utterance_id': turn['utterance_id'], **entry})
        return result

    def messages(self, sample):
        content = [{'type': 'text', 'text': prompt(sample, getattr(self.args, 'output_format', 'unconstrained'))}]
        clips = self.clips(sample)
        groups = ([clips] if clips else []) if self.args.model_kind == 'audio-flamingo-3' else (
            audio_groups(clips) if self.args.model_kind == 'voxtral-small-24b' else [[clip] for clip in clips])
        for group in groups:
            import soundfile as sf
            values_list, timeline = [], []
            samplerate, channels, frames = None, None, 0
            for turn, path, start, end in group:
                with sf.SoundFile(path) as audio:
                    if samplerate is None:
                        samplerate, channels = audio.samplerate, audio.channels
                    elif (audio.samplerate, audio.channels) != (samplerate, channels):
                        raise ValueError('Grouped audio must share samplerate/channels; no automatic resampling')
                    audio.seek(start)
                    values = audio.read(end-start, dtype='float32', always_2d=True)
                    values_list.append(values)
                    timeline.append(f"utterance {turn['index']+1}, speaker {turn['speaker']}: {frames/samplerate:.6f}-{(frames+len(values))/samplerate:.6f}s")
                    frames += len(values)
            if getattr(self.args, 'audio_layout', 'utterance') == 'dialogue_prefix':
                from dialogue_audio import prefix_label
                label = prefix_label(self, sample, group)
                if len(values_list) == 1:
                    values = values_list[0]
                else:
                    import numpy as np
                    values = np.concatenate(values_list, axis=0)
            elif len(group) == 1:
                values = values_list[0]
                label = f"Audio for utterance {turn['index']+1}, speaker {turn['speaker']} (segment starting at frame {start}):"
            else:
                import numpy as np
                values = np.concatenate(values_list, axis=0)
                label = 'Chronological visible-history audio, with original speakers. Segment timeline within this recording:\n'+'\n'.join(timeline)
            buf = io.BytesIO()
            sf.write(buf, values, samplerate, format='WAV', subtype='FLOAT')
            encoded = base64.b64encode(buf.getvalue()).decode()
            audio_content = ({'type': 'input_audio', 'input_audio': {'data': encoded, 'format': 'wav'}}
                             if self.args.model_kind == 'voxtral-small-24b' else
                             {'type': 'audio_url', 'audio_url': {'url': 'data:audio/wav;base64,'+encoded}})
            if self.args.model_kind in ('audio-flamingo-3', 'phi-4-multimodal-instruct') and getattr(self.args, 'output_format', 'unconstrained') == 'plain':
                label = '\n\n'+label+'\n'
            content.extend([{'type': 'text', 'text': label},
                            audio_content])
        # One user message works with both existing templates, without system-role differences.
        if self.args.model_kind == 'phi-4-multimodal-instruct' and clips:
            # Official Phi4MM speech format puts audio placeholders before the task instruction.
            content.append(content.pop(0))
        return [{'role': 'user', 'content': content}]

    def complete(self, sample):
        payload = {'model': self.args.served_model, 'messages': self.messages(sample),
                   'max_tokens': self.args.max_new_tokens, 'temperature': self.args.temperature,
                   'seed': self.args.seed}
        if getattr(self.args, 'output_format', 'unconstrained') == 'json-schema':
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': sample['task'], 'strict': True, 'schema': output_schema(sample)}}
        if self.args.model_kind in ('gemma4-e4b', 'qwen3-instruct'):
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        if self.args.condition != 'text' and self.args.model_kind == 'gemma4-e4b':
            payload['mm_processor_kwargs'] = {'audio_kwargs': {'padding': 'max_length', 'max_length': 480000}}
        return self.request('/chat/completions', payload)

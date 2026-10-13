"""Official Voxtral/Kimi text-only adapters; no model imports during dry-run."""
from pathlib import Path
import hashlib
import importlib.metadata
import time
from types import SimpleNamespace
from model_io import Client
from tasks import prompt


def normalize_kimi_features(kwargs):
    """The official generate API passes [] for text-only Whisper features.

    The checkpoint uses `is not None` to enter its audio branch, then reads
    features[0]. None is the no-audio sentinel expected by that forward path.
    Preserve real audio feature lists and every other argument unchanged.
    """
    normalized = dict(kwargs)
    features = normalized.get('continous_feature')
    if isinstance(features, list) and not features:
        normalized['continous_feature'] = None
    return normalized


def quiet_token_progress(iterable, *args, **kwargs):
    return iterable


def install_kimi_message_stop(model, api):
    """Treat Kimi's message-end control token as end-of-text generation.

    Keep the official sampler's chosen token/probabilities; signal the
    existing loop's EOS branch when it chooses a message boundary. Record
    the actual sampled boundary ID. Do not repair/trim JSON after generation.
    """
    stops = {model.extra_tokens.msg_end, model.extra_tokens.kimia_text_eos}
    original_sampler = api.KimiASampler
    class MessageStopSampler(original_sampler):
        def sample_text_logits(self, logits, recent_tokens=None):
            token = super().sample_text_logits(logits, recent_tokens=recent_tokens)
            token_id = token.item()
            if token_id in stops:
                model.voice_stop_token = token_id
                token.fill_(model.extra_tokens.kimia_text_eos)
            return token
    api.KimiASampler = MessageStopSampler
    model.voice_stop_token_ids = sorted(stops)
    model.voice_stop_token = None


class LocalClient(Client):
    def __init__(self, args):
        super().__init__(args)
        self.model = None

    def check_server(self):
        import torch
        if self.model is None:
            started = time.monotonic()
            print(f'[LOAD] {self.args.model_kind}: loading model', flush=True)
            if not torch.cuda.is_available():
                raise RuntimeError('Local inference requires a Slurm GPU allocation')
            torch.manual_seed(self.args.seed)
            if self.args.model_kind == 'voxtral-small-24b':
                from transformers import AutoProcessor, VoxtralForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(self.args.model, local_files_only=True)
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    self.args.model, torch_dtype=torch.bfloat16, device_map='auto', local_files_only=True).eval()
                self.context = min(32768, self.args.max_model_len or 32768)
            else:
                import json
                from huggingface_hub import snapshot_download
                manifest_path = Path(self.args.model)/'voice_download.json'
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    aux = Path(snapshot_download('THUDM/glm-4-voice-tokenizer', local_files_only=True))
                    if aux.name != manifest['audio_tokenizer_revision']:
                        raise RuntimeError('Cached Kimi audio tokenizer differs from setup revision')
                from kimia_infer.api import kimia
                # Scope this to the official API module; do not change global tqdm.
                kimia.tqdm = SimpleNamespace(tqdm=quiet_token_progress)
                KimiAudio = kimia.KimiAudio
                self.model = KimiAudio(self.args.model, load_detokenizer=False)
                self.model.alm.eval()
                install_kimi_message_stop(self.model, kimia)
                # Official text generator defaults to a total budget of 7500.
                self.context = min(7500, self.args.max_model_len or 7500)
                original = self.model._generate_loop
                def tracked_loop(*args, **kwargs):
                    audio, text = original(*args, **normalize_kimi_features(kwargs))
                    self.generated_tokens = len(text)
                    return audio, text
                self.model._generate_loop = tracked_loop
            print(f'[READY] {self.args.model_kind}: loaded in {time.monotonic()-started:.1f}s', flush=True)
        packages = {}
        for package in ('torch', 'transformers', 'mistral-common', 'flash-attn'):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        metadata = {'id': self.args.served_model, 'root': str(Path(self.args.model).resolve()),
                    'backend': 'local', 'context_limit': self.context, 'packages': packages}
        if self.args.model_kind == 'kimi-audio-7b-instruct':
            import inspect
            from kimia_infer.api.kimia import KimiAudio
            from kimia_infer.api.prompt_manager import KimiAPromptManager
            metadata['official_code_sha256'] = {
                cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                for cls in (KimiAudio, KimiAPromptManager)}
            metadata['stop_policy'] = 'voice-kimi-text-message-end-v1'
            metadata['stop_token_ids'] = self.model.voice_stop_token_ids
        return [metadata]

    def _check_length(self, length):
        if length + self.args.max_new_tokens > self.context:
            raise ValueError(f'Context overflow: {length} input + {self.args.max_new_tokens} output > '
                             f'{self.context}; no history was truncated. Use an explicit --context-turns for a new paired experiment.')

    def complete(self, sample):
        import torch
        with torch.inference_mode():
            if self.args.model_kind == 'voxtral-small-24b':
                messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt(sample)}]}]
                inputs = self.processor.apply_chat_template(messages)
                length = inputs.input_ids.shape[1]
                self._check_length(length)
                inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
                output = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens, do_sample=False)
                tokens = output[0, length:]
                text = self.processor.decode(tokens, skip_special_tokens=True)
                eos = self.model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                stopped = len(tokens) > 0 and tokens[-1].item() in eos
                count = len(tokens)
            else:
                self.model.voice_stop_token = None
                messages = [{'role': 'user', 'message_type': 'text', 'content': prompt(sample)}]
                history = self.model.prompt_manager.get_prompt(messages, output_type='text')
                length = history.to_tensor()[0].shape[1]
                self._check_length(length)
                _, text = self.model.generate(messages, output_type='text', text_temperature=0.0,
                    audio_temperature=0.0, text_repetition_penalty=1.0,
                    max_new_tokens=self.args.max_new_tokens)
                count = self.generated_tokens
                # Official loop excludes text EOS from its returned token count.
                stopped = self.model.voice_stop_token is not None
        response = {'model': self.args.served_model, 'choices': [{'message': {'role': 'assistant', 'content': text},
                'finish_reason': 'stop' if stopped else 'length'}],
                'usage': {'prompt_tokens': length, 'completion_tokens': count}}
        if self.args.model_kind == 'kimi-audio-7b-instruct':
            response['stop_policy'] = 'voice-kimi-text-message-end-v1'
            response['sampled_stop_token_id'] = self.model.voice_stop_token
        return response

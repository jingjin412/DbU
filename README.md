# DbU

A reproducible dialogue pipeline for MPDD and Bazinga: data preprocessing,
three dialogue tasks, and IndexTTS-2 audio generation.

| Component | Entry point |
| --- | --- |
| Data preprocessing | `datagen/` |
| Addressee recognition | `task/addressee_recognition.py` |
| Emotion recognition | `task/emotion_recognition.py` |
| Next-speaker ranking | `task/next_speaker_ranking.py` |
| Reference selection and audio generation | `audio/` |

**Neutral and every emotion use a fixed vector strength of 0.6.**
The vector activates exactly one class; `emo_alpha=1.0`, seed 42, and
`use_random=False`. Planning and synthesis reject incompatible settings,
including old 0.5 or per-class calibrated plans. Model inference temperature
is a separate setting (default 0).

## Setup

Use Python 3.10 or newer for CPU tools. IndexTTS setup creates its own
Python 3.10 environment and requires an allocated CUDA GPU for synthesis.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands below from the repository root.

## Data preprocessing

Obtain the MPDD source files (`dialogue_chinese.json`, `metadata.json`,
`readme.txt`) separately. Bazinga requires authorized Hugging Face access;
set `HF_TOKEN` in your environment if required. The downloader pins its
source revision and verifies checksums.

```bash
python datagen/fetch_bazinga.py
python datagen/prepare_data.py --mpdd-source /path/to/mpdd
python datagen/validate_data.py
python datagen/split_dialogues.py
python datagen/validate_split.py
```

See [data schema and provenance](data/README.md).

## Three tasks

Inspect prompts and samples without loading a model:

```bash
python task/run.py --task all --dataset mpdd --dry-run \
  --max-samples 10 --output-dir /tmp/voice-preview
```

For inference, first start a compatible model server in your GPU environment:

```bash
python task/run.py --task all --dataset mpdd --naming original \
  --model-kind gemma4-e4b --endpoint http://127.0.0.1:8901/v1 \
  --served-model voice-model --max-samples 10
```

See [task inputs, outputs, and scoring](task/README.md).

## Audio generation

Select reference voices, build neutral/predicted/gold plans, then synthesize
on Slurm. The complete commands and pinned runtime are in
[audio/README.md](audio/README.md).

## Validation

```bash
python -m unittest discover -s task -p 'test_*.py'
python -m unittest discover -s audio/index-TTS -p 'test_*.py'
```

Tests use temporary fixtures and do not require model weights or GPU inference.

## Repository scope

Only preprocessing, the three tasks, audio generation, their supporting code,
and small casting/review inputs are versioned. Dataset downloads, generated
WAVs, model weights, runtime caches, logs, and `results/` are ignored.
Result-analysis scripts, plots, probe experiments, and historical run reports
are excluded. Task predictions remain local outputs and can be supplied to
TTS planning; no existing results are bundled.

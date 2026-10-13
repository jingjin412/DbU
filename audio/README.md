# Audio generation

All commands start from the repo root. Reference selection uses the CPU
environment from `requirements.txt`. GPU synthesis uses the separate runtime
created by `setup.sbatch`; set your own Slurm account/partition on submission.

## Reference voices

```bash
python audio/select_voice/fetch_sources.py
python audio/select_voice/fetch_casting_supplement.py
python audio/select_voice/select_references.py
```

The source downloader verifies pinned AISHELL-3 files and fetches Bazinga
original audio. Optionally set `VOICE_AISHELL_CACHE` to an existing AISHELL-3
cache directory. MPDD casting preserves story/age variants and the first-story
seed; its small input tables are versioned under `select_voice/casting/`.
No source/reference WAVs are bundled. Voices selected automatically still need
acoustic review; age fallback is recorded rather than treated as age conversion.

For the reviewed Bazinga subset, rebuild the review candidates and apply the
included review decisions against verified source checksums:

```bash
python audio/select_voice/build_bazinga_review.py
python audio/select_voice/prepare_bazinga_round2.py
python audio/select_voice/finalize_bazinga_review.py
```

The HTML template is source; generated audition/review pages are ignored.

## Fixed 0.6 plans

Neutral and all seven emotion classes activate one vector component at 0.6,
with `emo_alpha=1.0`, seed 42 and `use_random=False`. The shared policy lives
in `index-TTS/emotion_settings.py`. Synthesis rejects incompatible old plans.

```bash
python audio/index-TTS/prepare_jobs.py --dataset mpdd \
  --output audio/index-TTS/plans/mpdd_neutral.json
python audio/index-TTS/prepare_jobs.py --dataset mpdd --min-participants 1 \
  --output audio/index-TTS/plans/mpdd_full_neutral.json
python audio/index-TTS/prepare_jobs.py --dataset bazinga \
  --output audio/index-TTS/plans/bazinga_neutral.json
python audio/select_voice/validate_references.py

python audio/index-TTS/prepare_jobs.py --dataset bazinga \
  --data data/bazinga/reference_qc_v1/dialogue.json \
  --voices audio/select_voice/manifests/bazinga_voices_reviewed.json \
  --output audio/index-TTS/plans/bazinga_reviewed_neutral.json
```

MPDD gold emotion:

```bash
python audio/index-TTS/prepare_gold_emotion.py
```

For predictions generated on the same dataset as casting, use the generic
planner. Provide an adjacent task `config.json`; missing/invalid predictions
block synthesis without a neutral/gold substitution:

```bash
python audio/index-TTS/prepare_jobs.py --dataset mpdd \
  --predictions /path/to/text-emotion/predictions.jsonl \
  --output audio/index-TTS/plans/mpdd_predicted.json
```

For model-specific predictions on the full source dataset and a reviewed
Bazinga casting subset, use the stricter join planner:

```bash
python audio/index-TTS/prepare_model_emotion.py --dataset bazinga \
  --model gemma4e4b --naming original \
  --predictions /path/to/text-emotion/predictions.jsonl \
  --output audio/index-TTS/plans/bazinga_gemma_predicted.json
```

Aliases are `gemma4e4b`, `qwen3omni`, `voxtral`. It verifies retained prompts,
turns, and source identities. Default `--invalid-policy error` blocks incomplete
plans; optional `skip` explicitly records exclusions and changes the sample set.
There are no hard-coded historical prediction run IDs.

## Setup and synthesis (Slurm)

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  audio/index-TTS/setup.sbatch
```

Setup clones IndexTTS-2 at `830f6f8f94a51fea23ab1d639027a86200075a4e`,
uses its frozen uv dependencies, downloads the model revision from
`versions.json`, and pins auxiliary repositories with `runtime_lock.json`.
`VOICE_BOOTSTRAP_PYTHON` can override the bootstrap Python executable.
Submit from the repo root, or export `VOICE_REPO_ROOT` explicitly.

Validate a plan without using a GPU:

```bash
python audio/index-TTS/synthesize.py \
  --jobs audio/index-TTS/plans/mpdd_neutral.json \
  --output-dir /tmp/voice-tts-check --validate-only --limit 0
```

After setup completes, synthesize a 10-utterance pilot:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  audio/index-TTS/run.sbatch plans/mpdd_neutral.json --limit 10
```

The jobs path is relative to `audio/index-TTS/` after the wrapper changes
directory; absolute paths also work. Use `--limit 0` for all utterances.
`--workers` controls parallel TTS workers. Outputs are under
`audio/index-TTS/runs/<job-id>/`; override with `VOICE_TTS_OUTPUT_DIR`.

Each run exports `audio_manifest.json`, keyed by stable utterance IDs,
for downstream task `--audio-manifest`. The cache includes text, reference,
emotion vector, runtime and source checksums. MPDD synthesis converts Traditional
Chinese to Simplified Chinese with OpenCC while preserving the source text.
Generated audio is pending quality review; generation does not establish
speaker identity or emotion accuracy.

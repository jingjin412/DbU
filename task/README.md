# Three dialogue tasks

Run `python task/run.py --help` for all options. Individual entry points
accept the same arguments and select their corresponding task.

| Task | Required response | Metrics |
| --- | --- | --- |
| Addressee recognition | `{"addressees":["P01","P02"]}` | Exact match, sample F1, micro F1 |
| Emotion recognition | `{"emotion":"neutral"}` | Accuracy/macro F1 when gold is available |
| Next-speaker ranking | `{"ranking":["P01","P02","P03"]}` | MRR, Hits@1, Hits@3 |

Emotion labels: happiness, angry, sadness, fear, disgust, surprise, neutral.
Bazinga has no gold emotion labels, so its emotion run records predictions
and invalid rate without reporting emotion accuracy.

Default inputs are `data/mpdd/dialogue.json` and
`data/bazinga/clean/dialogue.json`. MPDD gold emotions are read from
`data/mpdd/dialogue_chinese.json` for scoring. Use `--data` and `--metadata`
with a single `--dataset` to select a reviewed/custom subset.

The default participant threshold is three. `--context-turns 0` uses all
visible history through the current utterance. Ranking targets the next
transcript utterance's speaker, including the same speaker; the next utterance's
text/audio is excluded. Candidates use the full dialogue's speaker/listener
union, including membership of participants who appear later. Use
`--naming original` for source names or `anonymous` for P01-style IDs.

For audio, add `--condition neutral_audio`, `predicted_audio`, `gold_audio`,
or `real_audio`, and `--audio-manifest /path/to/audio_manifest.json`.
The manifest maps utterance IDs to WAV paths or objects with `path` and
optional provenance. Relative paths are resolved against the manifest folder.
The default audio scope includes all visible history; missing/overlong audio
fails preflight. Voxtral groups histories into at most five audio items.

```bash
python task/addressee_recognition.py --dataset mpdd --naming original \
  --model-kind qwen3-instruct --condition neutral_audio \
  --audio-manifest /path/to/audio_manifest.json --max-samples 10
```

The HTTP backend uses an existing OpenAI-compatible server. Set `VOICE_API_KEY`
in the environment if authentication is needed. `--model` records/verifies
its checkpoint path. Serving packages and weights are installed separately;
this repo does not launch servers or ship model weights. The optional local
text backend needs its model's own dependencies; Kimi also needs the official
source installed under `task/vendor/Kimi-Audio`.

Outputs go to `results/<task>/<dataset>/<model-kind>/<naming>/<run-id>/`:
config, raw replies, parsed predictions, and metrics. Use `--output-dir` to
change the root. Invalid answers stay in scoring denominators; transport
errors fail the run. `--resume --run-id <id>` requires identical inputs/settings.
Use a new run ID when changing conditions.

For model-predicted TTS, run emotion recognition under `--condition text`,
and pass its `predictions.jsonl` to the audio planner. The planner checks the
adjacent `config.json` and input checksums.

#!/usr/bin/env bash
#SBATCH --job-name=voice-indextts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=voice-tts-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
VOICE_REPO_ROOT="${VOICE_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
cd "$VOICE_REPO_ROOT/audio/index-TTS"
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export PATH="$PWD/runtime/bin:$PATH"
export NLTK_DATA="$PWD/cache/nltk_data"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
JOBS="${1:?Supply a prepared jobs JSON path}"
shift
test -x runtime/bin/python
test -f checkpoints/config.yaml
test "$(git -C vendor/index-tts rev-parse HEAD)" = 830f6f8f94a51fea23ab1d639027a86200075a4e
runtime/bin/python prepare_cache_layout.py
runtime/bin/python verify_offline_models.py
runtime/bin/python -u synthesize_parallel.py --jobs "$JOBS" --output-dir "${VOICE_TTS_OUTPUT_DIR:-runs/$SLURM_JOB_ID}" "$@"

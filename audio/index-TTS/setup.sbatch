#!/usr/bin/env bash
#SBATCH --job-name=voice-tts-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=voice-tts-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
VOICE_REPO_ROOT="${VOICE_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
cd "$VOICE_REPO_ROOT/audio/index-TTS"
BASE="$PWD"
export HF_HOME="$BASE/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export UV_CACHE_DIR="$BASE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$BASE/cache/python"
export UV_PROJECT_ENVIRONMENT="$BASE/runtime"
BOOTSTRAP="${VOICE_BOOTSTRAP_PYTHON:-python3}"
if [[ ! -x "$BASE/bootstrap/bin/uv" ]]; then
  "$BOOTSTRAP" -m venv "$BASE/bootstrap"
  "$BASE/bootstrap/bin/python" -m pip install uv==0.8.22
fi
REV=830f6f8f94a51fea23ab1d639027a86200075a4e
mkdir -p vendor
if [[ ! -d vendor/index-tts/.git ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --branch v2.0.0 --depth 1 https://github.com/index-tts/index-tts.git vendor/index-tts
fi
test "$(git -C vendor/index-tts rev-parse HEAD)" = "$REV"
cd vendor/index-tts
"$BASE/bootstrap/bin/uv" sync --frozen --python 3.10
"$BASE/bootstrap/bin/uv" pip install --python "$BASE/runtime/bin/python" opencc-python-reimplemented==0.1.7 imageio-ffmpeg==0.6.0
cd "$BASE"
# uv's managed Python may not find this cluster's system CA bundle.
# Use the installed trusted bundle for urllib/NLTK and later HTTPS downloads.
VOICE_TLS_CA_FILE=$("$BASE/runtime/bin/python" -c 'import certifi; print(certifi.where())')
export SSL_CERT_FILE="$VOICE_TLS_CA_FILE"
export REQUESTS_CA_BUNDLE="$VOICE_TLS_CA_FILE"
export NLTK_DATA="$BASE/cache/nltk_data"
"$BASE/runtime/bin/python" - <<'PYSETUP'
from pathlib import Path
import sys, imageio_ffmpeg, nltk
link=Path(sys.executable).parent/'ffmpeg'
if not link.exists():link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
for name in ('averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab'):
    if not nltk.download(name,quiet=True):raise RuntimeError('NLTK download failed: '+name)
PYSETUP
"$BASE/runtime/bin/python" download_models.py
"$BASE/runtime/bin/python" prepare_cache_layout.py
"$BASE/runtime/bin/python" verify_offline_models.py
"$BASE/runtime/bin/python" -m pip --version 2>/dev/null || true
"$BASE/bootstrap/bin/uv" pip freeze --python "$BASE/runtime/bin/python" > runtime_packages.txt
printf '%s\n' 'Setup completed; GPU inference has not been tested.'

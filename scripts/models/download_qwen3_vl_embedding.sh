#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_ID="${MODEL_ID:-qwen/Qwen3-VL-Embedding-2B}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/data/models/Qwen3-VL-Embedding-2B}"
RUNTIME_REPO="${RUNTIME_REPO:-https://github.com/QwenLM/Qwen3-VL-Embedding.git}"
RUNTIME_DIR="${RUNTIME_DIR:-$ROOT_DIR/data/models/Qwen3-VL-Embedding}"

python -m pip install -r "$ROOT_DIR/requirements.visual.txt"
python -m pip install modelscope
modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"

if [ ! -d "$RUNTIME_DIR/.git" ]; then
  rm -rf "$RUNTIME_DIR"
  git clone --depth 1 "$RUNTIME_REPO" "$RUNTIME_DIR"
else
  git -C "$RUNTIME_DIR" pull --ff-only
fi

cat <<EOF
Downloaded Qwen3-VL embedding runtime.

Recommended first-pass environment variables:

VISUAL_EMBEDDING_PROVIDER=sentence_transformers
VISUAL_EMBEDDING_MODEL=$MODEL_DIR
VISUAL_EMBEDDING_DIM=2048
MILVUS_COLLECTION_PREFIX=sightindex_qwen3vl

If you want to use the official Qwen runtime adapter instead of SentenceTransformers:

VISUAL_EMBEDDING_PROVIDER=qwen3_vl
QWEN3_VL_EMBEDDING_REPO_DIR=$RUNTIME_DIR
EOF

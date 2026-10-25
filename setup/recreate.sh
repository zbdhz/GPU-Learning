#!/usr/bin/env bash
# 通过精确版本记录创建新环境；不改动已有环境。
set -eo pipefail
setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v conda >/dev/null 2>&1; then
    echo '请先安装 Miniforge 并加载其 etc/profile.d/conda.sh，参见 doc/test-environment.md。' >&2
    exit 1
fi
conda_base="$(conda info --base)"
source "$conda_base/etc/profile.d/conda.sh"
gpu_env_name="${1:-gpu-dev-copy}"
if conda env list --json | "$conda_base/bin/python" -c 'import json, pathlib, sys; name = sys.argv[1]; sys.exit(0 if any(pathlib.Path(p).name == name for p in json.load(sys.stdin)["envs"]) else 1)' "$gpu_env_name"; then
    echo "环境已存在：$gpu_env_name。请指定一个新的环境名。" >&2
    exit 1
fi
conda create -y -n "$gpu_env_name" --file "$setup_dir/conda-linux-64.lock.txt"
conda activate "$gpu_env_name"
set -u
python -m pip install --index-url https://pypi.org/simple -r "$setup_dir/requirements-pip.lock.txt"
python "$setup_dir/link_nccl.py"
cp "$setup_dir/conda-pinned.txt" "$CONDA_PREFIX/conda-meta/pinned"
conda env config vars set -n "$gpu_env_name" \
    CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX" CUDACXX="$CONDA_PREFIX/bin/nvcc" \
    NCCL_HOME="$CONDA_PREFIX/opt/nccl" TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=4
echo "已创建 $gpu_env_name；请在当前终端执行 conda activate $gpu_env_name，再运行仓库中的 ./run_checks.sh。"

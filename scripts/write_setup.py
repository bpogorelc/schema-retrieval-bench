from pathlib import Path

lines = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "REPO=/mnt/c/Users/Bogdan/Documents/git/schema-retrieval-bench",
    "VENV=$HOME/.venvs/schema-bench",
    "cd \"$REPO\"",
    "rm -rf \"$REPO/.venv\" \"$VENV\"",
    "mkdir -p \"$HOME/.venvs\"",
    "python3 -c \"import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')\"",
    "python3 /tmp/get-pip.py --user",
    'export PATH="$HOME/.local/bin:$PATH"',
    "python3 -m pip install --user virtualenv",
    "# Keep the venv on the Linux filesystem; /mnt/c is very slow for many small files.",
    "python3 -m virtualenv \"$VENV\"",
    "# Do not symlink into the repo: Windows Git fails on WSL symlinks (open('.venv'): Invalid argument).",
    "source \"$VENV/bin/activate\"",
    "python -m pip install -U pip",
    "pip install torch --index-url https://download.pytorch.org/whl/cpu",
    "pip install sentence-transformers rank-bm25 numpy pandas pyarrow ray chdb pyyaml pytest",
    "pip install -e \"$REPO\" --no-deps",
    "python -c \"import chdb, ray, sentence_transformers, rank_bm25, torch; print('imports_ok', torch.__version__)\"",
    "",
]
path = Path(__file__).with_name("setup_wsl.sh")
path.write_bytes("\n".join(lines).encode("utf-8"))
print(f"wrote {path}")

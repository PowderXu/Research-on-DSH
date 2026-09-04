#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../../.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x "$project_root/evaluation/.venv/bin/python" ]]; then
  python_bin="$project_root/evaluation/.venv/bin/python"
else
  python_bin="python3"
fi

cd "$project_root"
PYTHONPATH="$project_root/evaluation${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$python_bin" -m dataset.scripts.prepare_unified_corpus "$@"

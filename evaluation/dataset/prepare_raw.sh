#!/usr/bin/env bash
set -euo pipefail

revision="c34e3dccad00f61133c799d20e7d1208a0e6cc92"
target="${1:-evaluation/dataset/raw/github-docs}"

if [[ -e "$target" && ! -d "$target/.git" ]]; then
  echo "Refusing to replace non-git path: $target" >&2
  exit 2
fi

if [[ ! -d "$target/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/github/docs.git "$target"
fi

git -C "$target" fetch --depth 1 origin "$revision"
git -C "$target" checkout --detach "$revision"

actual="$(git -C "$target" rev-parse HEAD)"
if [[ "$actual" != "$revision" ]]; then
  echo "Revision mismatch: expected $revision, got $actual" >&2
  exit 3
fi

echo "Prepared pinned GitHub Docs source at $target ($actual)"

#!/bin/sh
# Install the prepare-commit-msg hook into another repository:
#   scripts/install_hook.sh /path/to/your/repo
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
TARGET=$(cd "${1:?usage: install_hook.sh /path/to/repo}" && pwd)
HOOK="$TARGET/.git/hooks/prepare-commit-msg"
if [ -e "$HOOK" ]; then
  echo "a prepare-commit-msg hook already exists at $HOOK; not overwriting" >&2
  exit 1
fi
sed "s#__HERE__#$HERE#g" "$HERE/scripts/prepare-commit-msg" > "$HOOK"
chmod +x "$HOOK"
echo "installed $HOOK"

#!/usr/bin/env bash
# PostToolUse hook: after a .py file is edited or written, run the fast test suite.
# Scope: pytest only. It does not modify files, touch the network, or run anything else.
# Exit 2 sends the failure output back to Claude; exit 0 stays silent.
set -u

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
[ -d tests ] || exit 0

py=python
[ -x .venv/Scripts/python.exe ] && py=.venv/Scripts/python.exe
[ -x .venv/bin/python ] && py=.venv/bin/python

out=$(timeout 60 "$py" -m pytest -q -x --no-header 2>&1)
status=$?

# 0 = pass, 5 = no tests collected yet (fine early on)
if [ "$status" -ne 0 ] && [ "$status" -ne 5 ]; then
  printf '%s\n' "$out" | tail -n 25 >&2
  exit 2
fi
exit 0

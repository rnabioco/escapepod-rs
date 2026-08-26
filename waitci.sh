#!/bin/bash
# throwaway: block until a PR's checks are all terminal, then print them.
# Treats "no checks reported" as not-yet-created rather than as done.
pr="$1"
for _ in $(seq 1 60); do
  out=$(gh pr checks "$pr" 2>&1)
  if echo "$out" | grep -qE "	(pass|fail)	" && ! echo "$out" | grep -q "	pending	"; then
    echo "$out"
    exit 0
  fi
  sleep 45
done
echo "TIMED OUT waiting on #$pr"
gh pr checks "$pr" 2>&1
exit 1

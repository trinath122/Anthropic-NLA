#!/bin/sh
# Pre-push guard: run the secret + attribution scan with the first Python that
# actually works. On Windows, `python3` may be the Microsoft Store stub, so each
# candidate is tested before use; WSL's python3 is the last resort.
# If no interpreter works the push is blocked (fail closed).
for py in python3 python py; do
    if "$py" -c "import sys" >/dev/null 2>&1; then
        exec "$py" scripts/secret_scan.py
    fi
done
if command -v wsl >/dev/null 2>&1 && wsl -e python3 -c "import sys" >/dev/null 2>&1; then
    exec wsl -e python3 scripts/secret_scan.py
fi
echo "pre-push: no working Python found; push blocked (secret scan could not run)" >&2
exit 1

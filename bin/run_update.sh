#!/bin/sh
# Wrapper to execute update_data.py regardless of current working directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
/usr/bin/env python bin/update_data.py "$@"


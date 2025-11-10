#!/bin/sh
# Wrapper to execute run_execution.py from project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
/usr/bin/env python bin/run_execution.py "$@"


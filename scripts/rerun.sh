#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
for s in clean features models; do PYTHONPATH=src python -m readmit.pipeline $s; done

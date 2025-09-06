#! /usr/bin/env bash
VENV=$([[ -d venv ]] && echo venv || ([[ -d .venv ]] && echo .venv))
[[ -z $VENV ]] && echo "No venv found at 'venv', '.venv'!" && exit 1
source $VENV/bin/activate && accelerate-launch quickdif.py "$@"

#!/usr/bin/env bash

python sr.py --total_steps 300000 --swa_steps 50000 --version 9723 --angles --no_mmr --no_nan --no_eplusminus "$@"
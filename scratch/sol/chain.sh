#!/bin/sh
# chain.sh — h3c convrot build+repro+gates pipeline (writes scratch/sol/pipeline_status.json)
cd /Users/cyliu99/AI/h3c_mac_20261125/h3c || exit 1
exec nice -n 10 python3 scratch/sol/run_chain.py

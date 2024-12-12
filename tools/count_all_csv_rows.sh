#!/usr/bin/env bash

exp_name=$1
pred_path="/home/kaipoc/personal/research_vh/LLaVA-NeXT/outputs/miradata/pred_results/${exp_name}"

find "$pred_path" -type f -name '*.json' | while read -r jsonfile; do
    row_count=$(wc -l < "$jsonfile")
    echo "There are ${row_count} rows in ${jsonfile}"
done
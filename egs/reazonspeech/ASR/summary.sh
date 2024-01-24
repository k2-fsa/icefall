#!/bin/bash

summary_file="summary_results.txt"
> "$summary_file"

for file in wer-summary-valid-*-epoch-*-avg-*.txt; do
    # Extract epoch and avg values from the filename
    epoch=$(echo "$file" | grep -oP 'epoch-\K[0-9]+')
    avg=$(echo "$file" | grep -oP 'avg-\K[0-9]+')

    # Extract the WER value from the second line of the file
    wer=$(awk 'NR==2 {print $2}' "$file")

    # Append the results to the summary file
    echo "epoch $epoch avg $avg greedy_search $wer" >> "$summary_file"
done

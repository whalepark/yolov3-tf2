#!/bin/bash

timestamp=$(date +%Y%m%d_%H:%M:%S)
mkdir -p data/$timestamp/{1,5,10,15,20}

for filename in data/*; do
    if [[ -d $filename ]] || [[ ${filename: -4} == '.csv' ]]; then
        continue
    else
        rm -f $filename
    fi
done

for concurrent in 1 5 10 15 20; do
    for iter in $(seq 1 10); do
        bash exp_script.sh perf-shmem-rlimit -n=$concurrent
    done

    for filename in data/*; do
        if [[ -d $filename ]] || [[ ${filename: -4} == ".csv" ]]; then
            continue
        else
            mv $filename data/$timestamp/$concurrent
        fi
    done

    python perf_result_parser.py -d="data/$timestamp/${concurrent}" -i=10
done
#!/bin/bash

# exec 2>&1

python3.6 detect.py --image data/meme.jpg

start=$(date +%s.%N)
for i in $(seq 1 5); do
    python3.6 detect.py --image data/meme.jpg 2>&1 | grep inference_time &
done

wait
end=$(date +%s.%N)
elapsed=$(echo $end - $start | tr -d $'\t' | bc)
echo $elapsed
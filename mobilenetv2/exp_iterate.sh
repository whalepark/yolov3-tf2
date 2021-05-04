#!/bin/bash


mkdir -p data
me=$(whoami)
sudo chown -R $me data
sudo chgrp -R $me data


# commands=("latency-mon" "rusage-mon" "perf-mon" "latency" "rusage" "perf" "cprofile")
commands=("perf")


# for command in "${commands[@]}"; do
#     for i in 1 5 10; do
#         for j in 1 2 3 4 5 6 7 8 9 10; do
#             bash exp_script.sh $command -n=$i --ratio=0.5
#         done
#     done
# done

for command in "${commands[@]}"; do
    for i in 1 5 10; do
        for j in 1 2 3 4 5 6 7 8 9 10; do
            bash exp_script.sh $command -n=$i --ratio=0.8
        done
    done
done




# killall -9 pocketd
# sleep 3
# sudo ./pocket/pocketd &
# sleep 3
# bash ./exp_script.sh rusage -n=1

# killall -9 pocketd
# sleep 3
# sudo ./pocket/pocketd &
# sleep 3
# bash ./exp_script.sh rusage -n=5

# killall -9 pocketd
# sleep 3
# sudo ./pocket/pocketd &
# sleep 3
# bash ./exp_script.sh rusage -n=10


# killall -9 pocketd
# sleep 3
# sudo ./pocket/pocketd &
# sleep 3
# bash ./exp_script.sh cprofile -n=1

# killall -9 pocketd
# sleep 3
# sudo ./pocket/pocketd &
# sleep 3
# bash ./exp_script.sh cprofile -n=5

# killall -9 pocketd
# sleep 3
# sudo ./pocket/pocketd &
# sleep 3
# bash ./exp_script.sh cprofile -n=10


# killall -9 pocketd
# sleep 3
# sudo ./pocket/pocketd &
# sleep 3
# bash ./exp_script.sh perf -n=1

# killall -9 pocketd
# sleep 3
# sudo ./pocket/pocketd &
# sleep 3
# bash ./exp_script.sh perf -n=5

# killall -9 pocketd
# sleep 3
# sudo ./pocket/pocketd &
# sleep 3
# bash ./exp_script.sh perf -n=10

# exit



# for i in $(seq 1 4); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.5 &
#     sleep 3
#     bash ./exp_script.sh rusage -n=1
# done

# for i in $(seq 1 4); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.5 &
#     sleep 3
#     bash ./exp_script.sh rusage -n=5
# done

# for i in $(seq 1 4); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.5 &
#     sleep 3
#     bash ./exp_script.sh rusage -n=10
# done


# for i in $(seq 1 4); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.5 &
#     sleep 3
#     bash ./exp_script.sh cprofile -n=1
# done

# for i in $(seq 1 4); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.5 &
#     sleep 3
#     bash ./exp_script.sh cprofile -n=5
# done

# for i in $(seq 1 2); do
#     sudo ./pocket/pocketd --ratio=0.5 &
#     sleep 3
#     killall -9 pocketd
#     sleep 3
#     bash ./exp_script.sh cprofile -n=10
# done


# for i in $(seq 1 4); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.5 &
#     sleep 3
#     bash ./exp_script.sh perf -n=1
# done

# for i in $(seq 1 4); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.5 &
#     sleep 3
#     bash ./exp_script.sh perf -n=5
# done

# for i in $(seq 1 4); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.5 &
#     sleep 3
#     bash ./exp_script.sh perf -n=10
# done



# exit

echo 0.8

# for i in $(seq 1 9); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.8 &
#     sleep 3
#     bash ./exp_script.sh rusage -n=1
# done

# for i in $(seq 1 9); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.8 &
#     sleep 3
#     bash ./exp_script.sh rusage -n=5
# done

# for i in $(seq 1 9); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.8 &
#     sleep 3
#     bash ./exp_script.sh rusage -n=10
# done


# for i in $(seq 1 9); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.8 &
#     sleep 3
#     bash ./exp_script.sh perf -n=1
# done

# for i in $(seq 1 2); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.8 &
#     sleep 3
#     bash ./exp_script.sh perf -n=5
# done

# for i in $(seq 1 9); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.8 &
#     sleep 3
#     bash ./exp_script.sh perf -n=10
# done


# for i in $(seq 1 9); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.8 &
#     sleep 3
#     bash ./exp_script.sh cprofile -n=1
# done

# for i in $(seq 1 4); do
#     killall -9 pocketd
#     sleep 3
#     sudo ./pocket/pocketd --ratio=0.8 &
#     sleep 3
#     bash ./exp_script.sh cprofile -n=5
# done

# for i in $(seq 1 9); do
#     sudo ./pocket/pocketd --ratio=0.8 &
#     sleep 3
#     killall -9 pocketd
#     sleep 3
#     bash ./exp_script.sh cprofile -n=10
# done

# exit

# for i in $(seq 1 10); do
#     bash ./exp_script.sh motivation -n=1
# done
#!/bin/bash

BASEDIR=$(dirname $0)
cd $BASEDIR
NUMINSTANCES=1
TIMESTAMP=$(date +%Y%m%d-%H:%M:%S)
NETWORK=tf-grpc-exp

EXP_ROOT="${HOME}/settings/tf-slim/lightweight/pjt/grpc"
SUBNETMASK=111.222.0.0/16
SERVER_IP=111.222.3.26

source internal_functions.sh

function parse_arg() {
    for arg in $@; do
        case $arg in
            -n=*|--num=*)
                NUMINSTANCES="${arg#*=}"
                ;;
            -s=*|--server=*)
                SERVER="${arg#*=}"
                ;;
            -ri=*|--random-interval=*)
                INTERVAL="${arg#*=}"
                ;;
        esac
    done
}

function generate_rand_num() {
    if [[ "$INTERVAL" = "1" ]]; then
        local base=$(printf "0.%01d\n" $(( RANDOM % 1000 )))
        # echo ${#base}
        echo "$base * 3" | bc
    else
        echo 0
    fi
}

function init() {
    docker rm -f $(docker ps -a | grep "grpc_server\|grpc_app_\|grpc_exp_server\|grpc_exp_app_\|" | awk '{print $1}') > /dev/null 2>&1
    docker network rm $NETWORK
    docker network create --driver=bridge --subnet=$SUBNETMASK $NETWORK
}

function finalize() {
    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"
}

function health_check() {
    init
    # Run a server
    _run_d_server grpc_exp_server grpc_exp_server_00 $NETWORK 5

    # Run a client with hello
    _run_client grpc_exp_client grpc_exp_app_00 grpc_exp_server_00 $NETWORK python3.6 detect.py --hello
}

function health_check_dev() {
    init
    # Run a server
    _run_d_server_dev grpc_exp_server grpc_exp_server_00 $NETWORK 5

    # Run a client with hello
    _run_client_dev grpc_exp_client grpc_exp_app_00 grpc_exp_server_00 $NETWORK "bash -c \"git pull && python3.6 detect.py --hello && perf stat -p $! -e cycles,page-faults\""
}

function build_image() {
    docker rmi -f $(docker image ls | grep "grpc_exp_server\|grpc_exp_client" | awk '{print $1}')

    cp ../../yolov3.weights ./dockerfiles
    docker image build --no-cache -t grpc_exp_client -f dockerfiles/Dockerfile.idapp dockerfiles
    docker image build --no-cache -t grpc_exp_server -f dockerfiles/Dockerfile.idser dockerfiles

    # docker rmi -f grpc_exp_client
    # docker image build --no-cache -t grpc_exp_client -f dockerfiles/Dockerfile.idapp dockerfiles
}

function build_shmem() {
    docker rmi -f $(docker image ls | grep "grpc_exp_shmem_server\|grpc_exp_shmem_client" | awk '{print $1}')

    cp ../../yolov3.weights ./dockerfiles
    docker image build --no-cache -t grpc_exp_shmem_client -f dockerfiles/Dockerfile.shmem.idapp dockerfiles
    docker image build --no-cache -t grpc_exp_shmem_server -f dockerfiles/Dockerfile.shmem.idser dockerfiles

    # docker rmi -f grpc_exp_client
    # docker image build --no-cache -t grpc_exp_client -f dockerfiles/Dockerfile.idapp dockerfiles
}

function perf() {
    local numinstances=$1
    local events=$2
    local pid_list=()
    local container_list=()

    local server_container_name=grpc_exp_server_id_00
    local server_image=grpc_exp_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server ${server_image} ${server_container_name} $NETWORK 5

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_id_${index}

        # _run_client $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image data/meme.jpg"
        # _run_d_client $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image data/meme.jpg"
        _run_d_client $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "bash -c 'git pull && python3.6 detect.py --image data/photographer.jpg'"
        # _run_client grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image images/photographer.jpg"
        # _run_d_client $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image images/photographer.jpg"
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_id_${index}

        docker wait "${container_name}"
    done

    # For debugging
    docker logs grpc_exp_app_id_0001
    # docker logs grpc_exp_app_id_0004
    # exit

    # Baseline: Dockerfiles in ~/settings/lightweight must be built in advance before executing the below commands.
    server_container_name=grpc_exp_server_bin_00
    server_image=grpc_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server ${server_image} ${server_container_name} $NETWORK 5

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        # _run_client $i grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image data/meme.jpg"
        # _run_d_client $i grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image data/meme.jpg"
        _run_d_client $i grpc_client ${container_name} ${server_container_name} $NETWORK "bash -c 'git pull && python3.6 detect.py --image data/photographer.jpg'"
        # _run_client grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image images/photographer.jpg"
        # _run_d_client $i grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image images/photographer.jpg"
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        docker wait "${container_name}"
    done

    # For debugging
    docker logs grpc_exp_app_bin_0001
    # docker logs grpc_exp_app_bin_0004
    # exit

    init
}

function init_ramfs() {
    local dir="$(pwd)/ramfs/"
    echo "${dir}"

    sudo umount "${dir}"
    sudo rm -rf "${dir}"

    if [[ ! -d "${dir}" ]]; then
        sudo mkdir -p "${dir}"
        sudo mount --make-shared "${dir}"
        sudo mount --make-shared -t tmpfs -o size=100M tmpfs "${dir}"
        sudo cp ../images/* "${dir}"
        sudo find ../data/ ! -name yolov3.weights -exec cp -t "${dir}" {} +
    fi
}

function perf_ramfs() {
    local numinstances=$1
    local events=$2
    local pid_list=()
    local container_list=()

    local server_container_name=grpc_exp_server_id_00
    local server_image=grpc_exp_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server ${server_image} ${server_container_name} $NETWORK 5

    for i in $(seq 1 $numinstances); do
    # for((i=0; i < numinstances; ++i)); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_id_${index}

        # _run_client $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image /ramfs/meme.jpg"
        # _run_d_client_w_ramfs $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image /ramfs/meme.jpg"
        _run_d_client_w_ramfs $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "bash -c 'git pull && python3.6 detect.py --image /ramfs/photographer.jpg'"

        # _run_client grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image /ramfs/photographer.jpg"
        # _run_d_client_w_ramfs $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image /ramfs/photographer.jpg"
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"


    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_id_${index}
        docker wait "${container_name}"
    done

    # for debugging
    # docker logs grpc_exp_app_id_0001
    # docker logs grpc_exp_app_id_0004
    

    # Baseline: Dockerfiles in ~/settings/lightweight must be built in advance before executing the below commands.
    server_container_name=grpc_exp_server_bin_00
    server_image=grpc_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server ${server_image} ${server_container_name} $NETWORK 5

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        # _run_client $i grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image ramfs/meme.jpg"
        # _run_d_client_w_ramfs $i grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image /ramfs/meme.jpg"
        _run_d_client_w_ramfs $i grpc_client ${container_name} ${server_container_name} $NETWORK "bash -c 'git pull && python3.6 detect.py --image /ramfs/photographer.jpg'"

        # _run_client grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image ramfs/photographer.jpg"
        # _run_d_client_w_ramfs $i grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image /ramfs/photographer.jpg"
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        docker wait "${container_name}"
    done

    # For debugging
    # docker logs grpc_exp_app_bin_0001
    # docker logs grpc_exp_app_bin_0004
    # exit

    init
}

function perf_redis() {
    local numinstances=$1
    local events=$2
    local pid_list=()
    local container_list=()

    local server_container_name=grpc_exp_server_id_00
    local server_image=grpc_exp_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server_redis ${server_image} ${server_container_name} $NETWORK $SERVER_IP 5

    

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_id_${index}

        _run_d_client_w_redis $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK $SERVER_IP
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_id_${index}
        docker wait "${container_name}"
    done

    # for debugging
    docker logs grpc_exp_app_id_0001
    # docker logs grpc_exp_app_id_0004
    exit

    # Baseline: Dockerfiles in ~/settings/lightweight must be built in advance before executing the below commands.
    server_container_name=grpc_exp_server_bin_00
    server_image=grpc_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server_redis ${server_image} ${server_container_name} $NETWORK $SERVER_IP 5

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        _run_d_client_w_redis $i grpc_client ${container_name} ${server_container_name} $NETWORK $SERVER_IP "python3.6 detect.py --image /images/meme.jpg"
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        docker wait "${container_name}"
    done

    # For debugging
    docker logs grpc_exp_app_bin_0001
    # docker logs grpc_exp_app_bin_0004
    # exit

    init
    # sudo kill -9 $(ps aux | grep 'perf stat' | awk '{print $2}')
}


function perf_shmem() {
    local numinstances=$1
    local events=$2
    local pid_list=()
    local container_list=()

    local server_container_name=grpc_exp_server_shmem_00
    local server_image=grpc_exp_shmem_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server_shmem_dev ${server_image} ${server_container_name} $NETWORK 5

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_shmem_${index}

        # _run_d_client_shmem_dev $i grpc_exp_shmem_client ${container_name} ${server_container_name} $NETWORK 'python3.6 detect.py --object shmem --image data/street.jpg'
        _run_d_client_shmem_dev $i grpc_exp_shmem_client ${container_name} ${server_container_name} $NETWORK 'python3.6 detect.py --object shmem --image data/street.jpg'
        sleep $(generate_rand_num)
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_shmem_${index}

        docker wait "${container_name}"
    done

    # For debugging
    docker logs grpc_exp_app_shmem_0001
    # docker logs grpc_exp_app_id_0004
    # docker logs grpc_exp_server_shmem_00

    server_container_name=grpc_exp_server_bin_00
    # server_image=grpc_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server_shmem_dev ${server_image} ${server_container_name} $NETWORK 5

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        _run_d_client_shmem_dev $i grpc_exp_shmem_client ${container_name} ${server_container_name} $NETWORK 'python3.6 detect.py --object bin --image data/street.jpg'
        sleep $(generate_rand_num)
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        docker wait "${container_name}"
    done

    # For debugging
    docker logs grpc_exp_app_bin_0001
    # docker logs grpc_exp_app_bin_0004
    # exit

    init
}


function perf_shmem_rlimit() {
    local numinstances=$1
    local events=$2
    local pid_list=()
    local container_list=()

    local server_container_name=grpc_exp_server_shmem_00
    local server_image=grpc_exp_shmem_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server_shmem_rlimit ${server_image} ${server_container_name} $NETWORK 5

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_shmem_${index}

        # _run_d_client_shmem_dev $i grpc_exp_shmem_client ${container_name} ${server_container_name} $NETWORK 'python3.6 detect.py --object shmem --image data/street.jpg'
        _run_d_client_shmem_rlimit $i grpc_exp_shmem_client ${container_name} ${server_container_name} $NETWORK 'python3.6 detect.py --object shmem --image /img/photographer.jpg'
        sleep $(generate_rand_num)
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_shmem_${index}

        docker wait "${container_name}"
    done

    # For debugging
    docker logs grpc_exp_app_shmem_0001
    # docker logs grpc_exp_app_id_0004
    # docker logs grpc_exp_server_shmem_00

    server_container_name=grpc_exp_server_bin_00
    # server_image=grpc_server

    init
    sudo kill -9 $(ps aux | grep unix_multi | awk '{print $2}') > /dev/null 2>&1
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    sudo python unix_multi_server.py &
    _run_d_server_shmem_rlimit ${server_image} ${server_container_name} $NETWORK 5

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        # _run_d_client_shmem_rlimit $i grpc_exp_shmem_client ${container_name} ${server_container_name} $NETWORK 'python3.6 detect.py --object bin --image data/street.jpg'
        _run_d_client_shmem_rlimit $i grpc_exp_shmem_client ${container_name} ${server_container_name} $NETWORK 'python3.6 detect.py --object bin --image /img/photographer.jpg'
        sleep $(generate_rand_num)
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_bin_${index}

        docker wait "${container_name}"
    done

    # For debugging
    docker logs grpc_exp_app_bin_0001
    # docker logs grpc_exp_app_bin_0004
    # exit

    init
}

function cleanup_shm() {
    while IFS=$'\n' read -r line; do
        if [[ -z $line ]]; then
            continue
        fi
        semid=$(echo $line | awk '{print $2}')
        ipcrm -s $semid
    done <<< $(ipcs -s | tail -n +4)

    while IFS=$'\n' read -r line; do
        if [[ -z $line ]]; then
            continue
        fi
        shmid=$(echo $line | awk '{print $2}')
        ipcrm -m $shmid
    done <<< $(ipcs -m | tail -n +4)
}

function help() {
    echo Usage: ./exp_script.sh COMMAND [OPTIONS]
    echo Supported Commands:
    echo -e '\thealth, help, build, rtt, cpu, pfault, cache, tlb, ...'
    echo example: bash ./exp_script.sh health
    echo example: bash ./exp_script.sh rtt
}

trap finalize SIGINT
COMMAND=$([[ $# == 0 ]] && echo help || echo $1)
parse_arg ${@:2}

case $COMMAND in
    build)
        build_image
        ;;
    health|hello)
        health_check
        ;;
    perf)
        perf $NUMINSTANCES cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses
        ;;
    'perf-ramfs')
        init_ramfs
        perf_ramfs $NUMINSTANCES cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses
        ;;
    'perf-redis')
        perf_redis $NUMINSTANCES cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses
        ;;
    'perf-shmem')
        perf_shmem $NUMINSTANCES cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses
        ;;
    'perf-shmem-rlimit')
        perf_shmem_rlimit $NUMINSTANCES cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses
        ;;
    'build-shmem')
        build_shmem
        ;;
    'cleanup-shm')
        cleanup_shm
        ;;
    debug)
        init_ramfs
        ls ramfs
        ;;
    *|help)
        help
        ;;
esac
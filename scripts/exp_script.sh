#!/bin/bash

BASEDIR=$(dirname $0)
cd $BASEDIR
NUMINSTANCES=1
TIMESTAMP=$(date +%Y%m%d-%H:%M:%S)
NETWORK=tf-grpc-exp

EXP_ROOT="${HOME}/settings/tf-slim/lightweight/pjt/grpc"

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

function init() {
    docker rm -f $(docker ps -a | grep "grpc_server\|grpc_app_\|grpc_exp_server\|grpc_exp_app_\|" | awk '{print $1}') > /dev/null 2>&1
    docker network rm $NETWORK
    docker network create --driver=bridge $NETWORK
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
    # docker rmi -f $(docker ps -a | grep "grpc_exp_client" | awk '{print $1}')

    cp ../../yolov3.weights ./dockerfiles
    docker image build --no-cache -t grpc_exp_client -f dockerfiles/Dockerfile.idapp dockerfiles
    docker image build --no-cache -t grpc_exp_server -f dockerfiles/Dockerfile.idser dockerfiles
    # docker image build -t grpc_exp_server -f dockerfiles/Dockerfile.idser ${HOME}
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
        _run_d_client $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image data/meme.jpg"
        # _run_d_client $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "bash -c 'git pull && python3.6 detect.py --image data/meme.jpg'"
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
    # docker logs grpc_exp_app_id_0001
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
        _run_d_client $i grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image data/meme.jpg"
        # _run_d_client $i grpc_client ${container_name} ${server_container_name} $NETWORK "bash -c 'git pull && python3.6 detect.py --image data/meme.jpg'"
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
    # docker logs grpc_exp_app_bin_0001
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
        # sudo cp -r ! (yolov3.weights|../data/*) "${dir}"
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
        local index=$(printf "%04d" $i)
        local container_name=grpc_exp_app_id_${index}

        # _run_client $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image /ramfs/meme.jpg"
        _run_d_client_w_ramfs $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image /ramfs/meme.jpg"
        # _run_d_client_w_ramfs $i grpc_exp_client ${container_name} ${server_container_name} $NETWORK "bash -c 'git pull && python3.6 detect.py --image /ramfs/meme.jpg'"

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

        # _run_client $i grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image ramfs/meme.jpg"
        _run_d_client_w_ramfs $i grpc_client ${container_name} ${server_container_name} $NETWORK "python3.6 detect.py --image /ramfs/meme.jpg"
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

function compare_rtt() {
    init
    local grpc_rtt=$(_measure_rtt_grpc)

    # init
    # local grpc_rtt_path=$(_measure_rtt_grpc_w_path)

    echo grpc_rtt=${grpc_rtt}
    echo grpc_rtt_path=${grpc_rtt_path}
}

function compare_cpu_cycles() {
    init
    local grpc_rtt=$(_measure_cpu_cycles)

    # init
    # local grpc_rtt_path=$(_measure_cpu_cycles_w_path)

    echo grpc_rtt=${grpc_rtt}
    echo grpc_rtt_path=${grpc_rtt_path}
}

function compare_page_faults() {
    init
    local grpc_rtt=$(_measure_page_faults)

    init
    local grpc_rtt_path=$(_measure_page_faults_w_path)

    echo grpc_rtt=${grpc_rtt}
    echo grpc_rtt_path=${grpc_rtt_path}
}

function compare_cache_misses() {
    init
    local grpc_rtt=$(_measure_cache_misses)

    init
    local grpc_rtt_path=$(_measure_cache_misses_grpc_w_path)

    echo grpc_rtt=${grpc_rtt}
    echo grpc_rtt_path=${grpc_rtt_path}
}

function compare_tlb_misses() {
    init
    local grpc_rtt=$(_measure_tlb_misses_grpc)

    init
    local grpc_rtt_path=$(_measure_tlb_misses_grpc_w_path)

    echo grpc_rtt=${grpc_rtt}
    echo grpc_rtt_path=${grpc_rtt_path}
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
    rtt)
        compare_rtt
        ;;
    cpu)
        compare_cpu_cycles
        ;;
    pfault)
        compare_page_faults
        ;;
    cache|llc)
        compare_cache_misses
        ;;
    tlb)
        compare_tlb_misses
        ;;
    debug)
        init_ramfs
        ls ramfs
        ;;
    *|help)
        help
        ;;
esac
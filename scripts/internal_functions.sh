#!/bin/bash

source utils.sh

## Todo:
##  - remove SYS_ADMIN, IPC_LOCK

function _run_d_server() {
    local image=$1
    local container=$2
    local network=$3
    local pause=$([[ "$#" == 4 ]] && echo $4 || echo 5)
    docker run \
        -d \
        --privileged \
        --network=$network \
        --pid=host \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --name=$container \
        --workdir='/root/yolov3-tf2' \
        --env YOLO_SERVER=1 \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        --volume $(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        --volume=$(pwd)/ramfs:/ramfs \
        --cpuset-cpus=0 \
        $image \
        bash -c "git pull && python tfrpc/server/yolo_server.py" ## Todo: subtitue with the line below after debug
        # python tfrpc/server/yolo_server.py
    utils_attach_root $container
    sleep $pause
    echo 'Server bootup!'
}

# function _run_d_server_w_ramfs() {
#     local image=$1
#     local container=$2
#     local network=$3
#     local pause=$([[ "$#" == 4 ]] && echo $4 || echo 5)
#     docker run \
#         -d \
#         --privileged \
#         --network=$network \
#         --pid=host \
#         --cap-add SYS_ADMIN \
#         --cap-add IPC_LOCK \
#         --name=$container \
#         --workdir='/root/yolov3-tf2' \
#         --env YOLO_SERVER=1 \
#         --volume /var/run/docker.sock:/var/run/docker.sock \
#         --volume $(pwd)/data:/data \
#         --volume=$(pwd)/sockets:/sockets \
#         --cpuset-cpus=0 \
#         $image \
#         python tfrpc/server/yolo_server.py
#     utils_attach_root $container
#     sleep $pause
#     echo 'Server bootup!'
# }

function _run_client() {
    local index=$(($1 % $NUMCPU))
    local image_name=$2
    local container_name=$3
    local server_container=$4
    local network=$5
    local command=$6
    docker rm -f ${container_name} > /dev/null 2>&1

    local docker_cmd="docker run \
        --volume=$(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        --volume=$(pwd)/../images:/img \
        --network=${network} \
        --name=${container_name} \
        --workdir='/root/yolov3-tf2' \
        --env SERVER_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $server_container) \
        --env CONTAINER_ID=${container_name} \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --cpuset-cpus=${index} \
        ${image_name} \
        ${command}"
    eval $docker_cmd
    
    #python3.6 detect.py --image data/meme.jpg # default command
}

function _run_d_client() {
    local index=$(($1 % $NUMCPU))
    local image_name=$2
    local container_name=$3
    local server_container=$4
    local network=$5
    local command=$6
    docker rm -f ${container_name} > /dev/null 2>&1

    local docker_cmd="docker run \
        -d \
        --volume=$(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        --volume=$(pwd)/../images:/img \
        --network=${network} \
        --name=${container_name} \
        --workdir='/root/yolov3-tf2' \
        --env SERVER_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $server_container) \
        --env CONTAINER_ID=${container_name} \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --cpuset-cpus=${index} \
        ${image_name} \
        ${command}"
    eval $docker_cmd
    
    #python3.6 detect.py --image data/meme.jpg # default command
}

function _run_d_client_w_ramfs() {
    local index=$(($1 % $NUMCPU))
    local image_name=$2
    local container_name=$3
    local server_container=$4
    local network=$5
    local command=$6
    docker rm -f ${container_name} > /dev/null 2>&1

    local docker_cmd="docker run \
        -d \
        --volume=$(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        --volume=$(pwd)/../images:/img \
        --volume=$(pwd)/ramfs:/ramfs \
        --network=${network} \
        --name=${container_name} \
        --workdir='/root/yolov3-tf2' \
        --env SERVER_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $server_container) \
        --env CONTAINER_ID=${container_name} \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --cpuset-cpus=${index} \
        ${image_name} \
        ${command}"
    eval $docker_cmd
    
    #python3.6 detect.py --image data/meme.jpg # default command
}

function _run_d_server_dev() {
    local image=$1
    local container=$2
    local network=$3
    local pause=$([[ "$#" == 4 ]] && echo $4 || echo 5)
    docker run \
        -d \
        --privileged \
        --network=$network \
        --pid=host \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --name=$container \
        --workdir='/root/yolov3-tf2' \
        --env YOLO_SERVER=1 \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        --volume $(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        $image \
        bash -c "git pull && echo misun && python tfrpc/server/yolo_server.py"
    utils_attach_root $container
    sleep $pause
    echo 'Server bootup!'
}

function _run_client_dev() {
    local image_name=$1
    local container_name=$2
    local server_container=$3
    local network=$4
    local command=$5
    docker rm -f ${container_name} > /dev/null 2>&1

    local docker_cmd="docker run \
        --volume=$(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        --volume=$(pwd)/../images:/img \
        --network=${network} \
        --name=${container_name} \
        --workdir='/root/yolov3-tf2' \
        --env SERVER_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $server_container) \
        --env CONTAINER_ID=${container_name} \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        ${image_name} \
        ${command}"
    eval $docker_cmd
    
    #python3.6 detect.py --image data/meme.jpg # default command
}

function _measure_rtt_grpc() {
    local rtt=''
    _run_d_server grpc_exp_server grpc_exp_server_00 $NETWORK 5
    docker ps
exit
    _run_client grpc_exp_client grpc_exp_app_00 grpc_exp_server_00 $NETWORK "bash -c \"git pull && python3.6 detect.py --rtt --hello && perf stat -p \$! -e cycles,page-faults\""

    # _run_client grpc_exp_client grpc_exp_app_00 grpc_exp_server_00 $NETWORK "bash -c \"git pull && python3.6 detect.py --rtt --echo misun\""

    # _run_client grpc_exp_client grpc_exp_app_00 grpc_exp_server_00 $NETWORK "bash -c \"git pull && python3.6 detect.py --rtt --integer\""

    # _run_client grpc_exp_client grpc_exp_app_00 grpc_exp_server_00 $NETWORK "bash -c \"git pull && python3.6 detect.py --rtt --image data/meme.jpg\""

    # _run_client grpc_exp_client grpc_exp_app_00 grpc_exp_server_00 $NETWORK "bash -c \"git pull && python3.6 detect.py --rtt --image images/photographer.jpg\""

    echo $rtt
}

function _measure_rtt_grpc_w_path() {
    local rtt=''
    echo $rtt
}

function _measure_cpu_cycles() {
    local rtt=''
    echo $rtt
}

function _measure_cpu_cycles_w_path() {
    local rtt=''
    echo $rtt
}

function _measure_page_faults() {
    local rtt=''
    echo $rtt
}

function _measure_page_faults_w_path() {
    local rtt=''
    echo $rtt
}

function _measure_cache_misses() {
    local rtt=''
    echo $rtt
}

function _measure_cache_misses_grpc_w_path() {
    local rtt=''
    echo $rtt
}

function _measure_tlb_misses_grpc() {
    local rtt=''
    echo $rtt
}

function _measure_tlb_misses_grpc_w_path() {
    local rtt=''
    echo $rtt
}
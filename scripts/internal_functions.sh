#!/bin/bash

source utils.sh

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
        $image \
        bash -c "git pull && perf stat -B -r 1 -e cycles,cache-misses -o /data/grpc_server_id_only.log --append python tfrpc/server/yolo_server.py"
    utils_attach_root $container
    sleep $pause
    echo 'Server bootup!'
}

function _run_client() {
    local image_name=$1
    local container_name=$2
    local server_container=$3
    local network=$4
    local command=$5
    docker rm -f ${container_name} > /dev/null 2>&1
    docker run \
        --volume=$(pwd)/data:/data \
        --network=${network} \
        --name=${container_name} \
        --workdir='/root/yolov3-tf2' \
        --env SERVER_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $server_container) \
        ${image_name} \
        ${command}
    
    #python3.6 detect.py --image data/meme.jpg # default command
}
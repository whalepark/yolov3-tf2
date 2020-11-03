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
        --volume=/var/lib/docker/overlay2:/layers \
        --volume=$(pwd)/ramfs:/ramfs \
        --cpuset-cpus=0 \
        $image \
        python tfrpc/server/yolo_server.py
        # bash -c "git pull && python tfrpc/server/yolo_server.py" ## Todo: subtitue with the line below after debug
        # --mount type=bind,source=/var/lib/docker/overlay2,target=/layers,bind-propagation=rshared \
    # utils_attach_root $container # It is mount-binded through docker args.
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
    
    local pid=$(docker inspect -f '{{.State.Pid}}' $container_name)
    sudo perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -p $pid -o ./data/perf_stat_${container_name}.log &
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
    
    local pid=$(docker inspect -f '{{.State.Pid}}' $container_name)
    sudo perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -p $pid -o ./data/perf_stat_${container_name}.log &
    #python3.6 detect.py --image data/meme.jpg # default command
}

function _run_d_client_shmem_dev() {
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
        --volume=$(pwd)/..:/root/yolov3-tf2 \
        --network=${network} \
        --name=${container_name} \
        --workdir='/root/yolov3-tf2' \
        --env SERVER_ADDR=${SERVER_IP} \
        --env CONTAINER_ID=${container_name} \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --cpuset-cpus=${index} \
        --ipc=container:${server_container} \
        ${image_name} \
        ${command}"
    eval $docker_cmd
    
    # local pid=$(docker inspect -f '{{.State.Pid}}' $container_name)
    # sudo perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -p $pid -o ./data/perf_stat_${container_name}.log &
    #python3.6 detect.py --image data/meme.jpg # default command
}

function _run_d_client_shmem_rlimit() {
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
        --volume=$(pwd)/..:/root/yolov3-tf2 \
        --memory=256mb \
        --cpus=1 \
        --cpuset-cpus=${index} \
        --network=${network} \
        --name=${container_name} \
        --workdir='/root/yolov3-tf2' \
        --env SERVER_ADDR=${SERVER_IP} \
        --env CONTAINER_ID=${container_name} \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --ipc=container:${server_container}   \
        ${image_name} \
        ${command}"
    eval $docker_cmd
    
    # local pid=$(docker inspect -f '{{.State.Pid}}' $container_name)
    # sudo perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -p $pid -o ./data/perf_stat_${container_name}.log &
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

    # local pid=$(docker inspect -f '{{.State.Pid}}' $container_name)
    # sudo perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -p ${pid} -o ./data/perf_stat_"${container_name}".log &
    #python3.6 detect.py --image data/meme.jpg # default command
}

function _run_d_client_w_redis() {
    local index=$(($1 % $NUMCPU))
    local image_name=$2
    local container_name=$3
    local server_container=$4
    local network=$5
    local server_ip=$6

    docker rm -f ${container_name} > /dev/null 2>&1

    docker run \
        -d \
        --volume=$(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        --volume=$(pwd)/../images:/images \
        --volume=$(pwd)/ramfs:/ramfs \
        --volume=$HOME/yolov3-tf2:/root/yolov3-tf2 \
        --network=${network} \
        --name=${container_name} \
        --workdir='/root/yolov3-tf2' \
        --env SERVER_ADDR=${server_ip} \
        --env CONTAINER_ID=${container_name} \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --cpuset-cpus=${index} \
        ${image_name} \
        python3.6 detect.py --image /images/meme.jpg --object redis
        # python3.6 detect.py --image /images/meme.jpg --object path

    # local pid=$(docker inspect -f '{{.State.Pid}}' $container_name)
    # sudo perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -p ${pid} -o ./data/perf_stat_"${container_name}".log &
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
        --ip=$SERVER_IP \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        --volume $(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        --volume=$(pwd)/ramfs:/ramfs \
        --volume=$(pwd)/../images:/images \
        --volume=$HOME/yolov3-tf2:/root/yolov3-tf2 \
        $image \
        python tfrpc/server/yolo_server.py
    utils_attach_root $container
    sleep $pause
    echo 'Server bootup!'
}

function _run_d_server_shmem_dev() {
    local image=$1
    local container=$2
    local network=$3
    local pause=$([[ "$#" == 4 ]] && echo $4 || echo 5)
    docker run \
        -d \
        --privileged \
        --network=$network \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --name=$container \
        --workdir='/root/yolov3-tf2' \
        --env YOLO_SERVER=1 \
        --ip=$SERVER_IP \
        --ipc=shareable \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        --volume $(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        --volume=/var/lib/docker/overlay2:/layers \
        --volume=$(pwd)/ramfs:/ramfs \
        --volume=$(pwd)/..:/root/yolov3-tf2 \
        --cpuset-cpus=0 \
        $image \
        python tfrpc/server/yolo_server.py
        # bash -c "git pull && python tfrpc/server/yolo_server.py" ## Todo: subtitue with the line below after debug
        # --mount type=bind,source=/var/lib/docker/overlay2,target=/layers,bind-propagation=rshared \
    # utils_attach_root $container # It is mount-binded through docker args.
    sleep $pause
    echo 'Server bootup!'
}

function _run_d_server_shmem_rlimit() {
    local image=$1
    local container=$2
    local network=$3
    local pause=$([[ "$#" == 4 ]] && echo $4 || echo 5)
        # --cpuset-cpus=0 \
    docker run \
        -d \
        --privileged \
        --network=$network \
        --cap-add SYS_ADMIN \
        --cap-add IPC_LOCK \
        --name=$container \
        --workdir='/root/yolov3-tf2' \
        --env YOLO_SERVER=1 \
        --ip=$SERVER_IP \
        --ipc=shareable \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        --volume $(pwd)/data:/data \
        --volume=$(pwd)/sockets:/sockets \
        --volume=/var/lib/docker/overlay2:/layers \
        --volume=$(pwd)/ramfs:/ramfs \
        --volume=$(pwd)/..:/root/yolov3-tf2 \
        --volume=$(pwd)/../images:/img \
        $image \
        python tfrpc/server/yolo_server.py
        # bash -c "git pull && python tfrpc/server/yolo_server.py" ## Todo: subtitue with the line below after debug
        # --mount type=bind,source=/var/lib/docker/overlay2,target=/layers,bind-propagation=rshared \
    # utils_attach_root $container # It is mount-binded through docker args.
    sleep $pause
    echo 'Server bootup!'
}

function _run_d_server_redis() {
    local image=$1
    local container=$2
    local network=$3
    local server_ip=$4
    local pause=$([[ "$#" == 5 ]] && echo $5 || echo 5)
    docker run \
        -d \
        --privileged \
        --network=$network \
        --ip=$server_ip \
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
        --volume=$(pwd)/../images:/images \
        --volume=$HOME/yolov3-tf2:/root/yolov3-tf2 \
        $image \
        bash -c "redis-server --daemonize yes && python tfrpc/server/yolo_server.py"
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

#!/bin/bash

BASEDIR=$(dirname $0)
cd $BASEDIR
NUMINSTANCES=1
TIMESTAMP=$(date +%Y%m%d-%H:%M:%S)
INTERVAL=0
RSRC_RATIO=0.5

SUBNETMASK=111.222.0.0/16
SERVER_IP=111.222.3.26


mkdir -p data

function parse_arg(){
    for arg in $@; do
        case $arg in
            -n=*|--num=*)
                NUMINSTANCES="${arg#*=}"
                ;;
            -s=*|--server=*)
                INTERVAL="${arg#*=}"
                ;;
            -ri=*|--random-interval=*)
                INTERVAL="${arg#*=}"
                ;;
        esac
    done
}

function util_get_running_time() {
    local container_name=$1
    local start=$(docker inspect --format='{{.State.StartedAt}}' $container_name | xargs date +%s.%N -d)
    local end=$(docker inspect --format='{{.State.FinishedAt}}' $container_name | xargs date +%s.%N -d)
    local running_time=$(echo $end - $start | tr -d $'\t' | bc)

    echo $running_time
}

function generate_rand_num() {
    local upto=$1
    if [[ "$INTERVAL" = "1" ]]; then
        local base=$(printf "0.%01d\n" $(( RANDOM % 1000 )))
        # echo ${#base}
        echo "${base} * ${upto}" | bc
    else
        echo 1.5
    fi
}

function run_server_basic() {
    local server_container_name=$1
    local server_ip=$2
    local server_image=$3
    docker run \
        -d \
        --privileged \
        --name=$server_container_name \
        --workdir='/root' \
        --env YOLO_SERVER=1 \
        --ip=$server_ip \
        --ipc=shareable \
        --cpus=1.0 \
        --memory=1024mb \
        --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
        --volume $(pwd)/data:/data \
        --volume=$(pwd)/../scripts/sockets:/sockets \
        --volume=$(pwd)/../tfrpc/server:/root/tfrpc/server \
        --volume=$(pwd)/../yolov3-tf2:/root/yolov3-tf2 \
        $server_image \
        python tfrpc/server/yolo_server.py
}

function run_server_cProfile() {
    local server_container_name=$1
    local server_ip=$2
    local server_image=$3
    local timestamp=$4
    local numinstances=$5
    docker run \
        -d \
        --privileged \
        --name=$server_container_name \
        --workdir='/root' \
        --env YOLO_SERVER=1 \
        --ip=$server_ip \
        --ipc=shareable \
        --cpus=1.0 \
        --memory=1024mb \
        --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
        --volume $(pwd)/data:/data \
        --volume=$(pwd)/../scripts/sockets:/sockets \
        --volume=$(pwd)/../tfrpc/server:/root/tfrpc/server \
        --volume=$(pwd)/../yolov3-tf2:/root/yolov3-tf2 \
        $server_image \
        python -m cProfile -o /data/${timestamp}-${numinstances}-cprofile/${server_container_name}.cprofile tfrpc/server/yolo_server.py
}

function run_server_perf() {
    local server_container_name=$1
    local server_ip=$2
    local server_image=$3
    docker run \
        -d \
        --privileged \
        --name=$server_container_name \
        --workdir='/root' \
        --env YOLO_SERVER=1 \
        --ip=$server_ip \
        --ipc=shareable \
        --cpus=1.0 \
        --memory=1024mb \
        --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
        --volume $(pwd)/data:/data \
        --volume=$(pwd)/../scripts/sockets:/sockets \
        --volume=$(pwd)/../tfrpc/server:/root/tfrpc/server \
        --volume=$(pwd)/../yolov3-tf2:/root/yolov3-tf2 \
        $server_image \
        python tfrpc/server/yolo_server.py
}

function init() {
    docker rm -f $(docker ps -a | grep "grpc_server\|grpc_app_\|grpc_exp_server\|grpc_exp_app\|pocket" | awk '{print $1}') > /dev/null 2>&1
    # docker network rm $NETWORK
    # docker network create --driver=bridge --subnet=$SUBNETMASK $NETWORK
}

function help() {
    echo help!!!!!!!
}

function build_docker_files() {
    # docker rmi -f $(docker image ls | grep "grpc_exp_shmem_server\|grpc_exp_shmem_client\|pocket" | awk '{print $1}')

    docker rmi -f pocket-mobilenet-monolithic-perf 
    # docker rmi -f pocket-mobilenet-server pocket-mobilenet-application pocket-mobilenet-monolithic 
    # docker image build --no-cache -t pocket-mobilenet-server -f dockerfiles/Dockerfile.pocket.ser dockerfiles
    # docker image build --no-cache -t pocket-mobilenet-application -f dockerfiles/Dockerfile.pocket.app dockerfiles
    # docker image build --no-cache -t pocket-mobilenet-perf-application -f dockerfiles/Dockerfile.pocket.perf.app dockerfiles
    # docker image build --no-cache -t pocket-mobilenet-monolithic -f dockerfiles/Dockerfile.monolithic dockerfiles
    docker image build --no-cache -t pocket-mobilenet-monolithic-perf -f dockerfiles/Dockerfile.monolithic.perf dockerfiles
}

function run_monolithic() {
    local numinstances=$1
    local rusage_logging_dir=$(realpath data/${TIMESTAMP}-latency)

    mkdir -p ${rusage_logging_dir}
    docker rm -f $(docker ps -a | grep "pocket\|monolithic" | awk '{print $1}') > /dev/null 2>&1    docker network rm $NETWORK

    docker run \
        --name mobilenetv2-monolithic-0000 \
        --cpus=1 \
        --memory=512mb \
        --volume=$(pwd)/data:/data \
        --volume=$(pwd):/root/mobilenetv2 \
        --volume=/home/cc/COCO/val2017:/root/coco2017 \
        --workdir=/root/mobilenetv2 \
        pocket-mobilenet-monolithic \
        python3 app.monolithic.py

    running_time=$(util_get_running_time mobilenetv2-monolithic-0000)
    echo $running_time > "${rusage_logging_dir}"/mobilenetv2-monolithic-0000.latency

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}

        docker \
            run \
                -d \
                --name ${container_name} \
                --cpus=1 \
                --memory=512mb \
                --volume=$(pwd)/data:/data \
                --volume=$(pwd):/root/mobilenetv2 \
                --volume=/home/cc/COCO/val2017:/root/coco2017 \
                --workdir=/root/mobilenetv2 \
                pocket-mobilenet-monolithic \
                python3 app.monolithic.py
        sleep $(generate_rand_num 3)
    done

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}

        docker wait "${container_name}"
        running_time=$(util_get_running_time "${container_name}")
        echo $running_time > "${rusage_logging_dir}"/"${container_name}".latency
        echo $running_time
    done

    # For debugging
    docker logs -f mobilenetv2-monolithic-$(printf "%04d" $numinstances)

}


function run_pocket() {
    local numinstances=$1
    local container_list=()
    local rusage_logging_dir=$(realpath data/${TIMESTAMP}-latency)
    local rusage_logging_file=tmp-service.log

    local server_container_name=pocket-server-001
    local server_image=pocket-mobilenet-server

    mkdir -p ${rusage_logging_dir}
    init

    docker run \
        -d \
        --privileged \
        --name=$server_container_name \
        --workdir='/root' \
        --env YOLO_SERVER=1 \
        --ip=$SERVER_IP \
        --ipc=shareable \
        --cpus=2.0 \
        --memory=2048mb \
        --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
        --volume $(pwd)/data:/data \
        --volume=$(pwd)/../scripts/sockets:/sockets \
        --volume=$(pwd)/../tfrpc/server:/root/tfrpc/server \
        $server_image \
        python tfrpc/server/yolo_server.py

    exit

    ../scripts/pocket/pocket \
        run \
            --measure-latency $rusage_logging_dir \
            -b pocket-mobilenet-application \
            -t pocket-client-0000 \
            -s ${server_container_name} \
            --memory=1024mb \
            --cpus=1 \
            --volume=$(pwd):/root/mobilenetv2 \
            --volume=$(pwd)/data:/data \
            --volume=$(pwd)/../scripts/sockets:/sockets \
            --volume=$(pwd)/../yolov3-tf2/images:/img \
            --volume=$(pwd)/../tfrpc/client:/root/client \
            --env CONTAINER_ID=pocket-app-000 \
            --workdir='/root/mobilenetv2' \
            -- python app.pocket.py

    exit

    ../scripts/pocket/pocket \
        wait \
        pocket-client-0000

    exit

    sudo ../scripts/pocket/pocket \
        rusage \
        init ${server_container_name} --dir ${rusage_logging_dir} 



    local start=$(date +%s.%N)
    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}

        # _run_d_client_shmem_rlimit $i grpc_exp_shmem_client ${container_name} ${server_container_name} $NETWORK 'python3.6 detect.py --object path --image data/street.jpg'
        # _run_d_client_shmem_rlimit $i grpc_exp_shmem_client ${container_name} ${server_container_name} $NETWORK 'python3.6 detect.py --object shmem --image /img/photographer.jpg'
        ../scripts/pocket/pocket \
                run \
                    -d \
                    --rusage $rusage_logging_dir \
                    -b pocket-mobilenet-application \
                    -t ${container_name} \
                    -s ${server_container_name} \
                    -n $NETWORK \
                    --memory=512mb \
                    --cpus=1 \
                    --volume=$(pwd)/data:/data \
                    --volume=$(pwd)/sockets:/sockets \
                    --volume=$(pwd)/../yolov3-tf2/images:/img \
                    --volume=$(pwd)/../yolov3-tf2:/root/yolov3-tf2 \
                    --volume=$(pwd)/../tfrpc/client:/root/tfrpc/client \
                    --env SERVER_ADDR=${SERVER_IP} \
                    --env CONTAINER_ID=${container_name} \
                    --workdir='/root/yolov3-tf2' \
                    -- python3.6 detect.py --object path --image data/street.jpg
        interval=$(generate_rand_num 3)
        echo interval $interval
        sleep $interval
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}

        # docker wait "${container_name}"
        ../scripts/pocket/pocket \
            wait \
                ${container_name}

    done

    ../scripts/pocket/pocket \
        rusage \
        measure ${server_container_name} --dir ${rusage_logging_dir} 

    local end=$(date +%s.%N)
    local elapsed_time=$(echo $end - $start | tr -d $'\t' | bc)
    echo shmem $numinstances $start $end $elapsed_time >> data/end-to-end

    # For debugging
    docker logs pocket-server-001
    docker logs -f pocket-client-$(printf "%04d" $numinstances)
    # docker ps -a
    # ls /sys/fs/cgroup/memory/docker/
}

function measure_latency() {
    local numinstances=$1
    local rusage_logging_dir=$(realpath data/${TIMESTAMP}-${numinstances}-latency)

    local server_container_name=pocket-server-001
    local server_image=pocket-mobilenet-server

    mkdir -p ${rusage_logging_dir}
    init

    run_server_basic $server_container_name $SERVER_IP $server_image
    sleep 3

    ../scripts/pocket/pocket \
        run \
            --measure-latency $rusage_logging_dir \
            -d \
            -b pocket-mobilenet-application \
            -t pocket-client-0000 \
            -s ${server_container_name} \
            --memory=512mb \
            --cpus=1 \
            --volume=$(pwd)/data:/data \
            --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
            --volume=$(pwd)/../tfrpc/client:/root/tfrpc/client \
            --volume=$(pwd):/root/mobilenetv2 \
            --volume=/home/cc/COCO/val2017:/root/coco2017 \
            --env RSRC_REALLOC_RATIO=${RSRC_RATIO} \
            --env CONTAINER_ID=pocket-client-0000 \
            --workdir='/root/mobilenetv2' \
            -- python3 app.pocket.py

    sleep 5


    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}

        ../scripts/pocket/pocket \
            run \
                --measure-latency $rusage_logging_dir \
                -d \
                -b pocket-mobilenet-application \
                -t ${container_name} \
                -s ${server_container_name} \
                --memory=512mb \
                --cpus=1 \
                --volume=$(pwd)/data:/data \
                --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
                --volume=$(pwd)/../tfrpc/client:/root/tfrpc/client \
                --volume=$(pwd):/root/mobilenetv2 \
                --volume=/home/cc/COCO/val2017:/root/coco2017 \
                --env RSRC_REALLOC_RATIO=${RSRC_RATIO} \
                --env CONTAINER_ID=${container_name} \
                --workdir='/root/mobilenetv2' \
                -- python3 app.pocket.py &
        interval=$(generate_rand_num 3)
        echo interval $interval
        sleep $interval
    done

    wait

    local folder=$(realpath data/${TIMESTAMP}-${numinstances}-graph)
    mkdir -p $folder
    for i in $(seq 0 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}
        docker logs $container_name 2>&1 | grep "graph_construction_time" > $folder/$container_name.graph
    done

    folder=$(realpath data/${TIMESTAMP}-${numinstances}-inf)
    mkdir -p $folder
    for i in $(seq 0 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}
        docker logs $container_name 2>&1 | grep "inference_time" > $folder/$container_name.inf
    done

    # For debugging
    docker logs pocket-server-001
    docker logs -f pocket-client-$(printf "%04d" $numinstances)
}

function measure_latency_monolithic() {
    local numinstances=$1
    local container_list=()
    local rusage_logging_dir=$(realpath data/${TIMESTAMP}-${numinstances}-latency-monolithic)
    local rusage_logging_file=tmp-service.log

    mkdir -p ${rusage_logging_dir}
    init


    # 512mb, oom
    # 512 + 256 = 768mb, oom
    # 1024mb, ok
    # 1024 + 256 = 1280mb
    # 1024 + 512 = 1536mb
    # 1024 + 1024 = 2048mb
    docker run \
        --name mobilenetv2-monolithic-0000 \
        --cpus=1 \
        --memory=512mb \
        --volume=$(pwd)/data:/data \
        --volume=$(pwd):/root/mobilenetv2 \
        --volume=/home/cc/COCO/val2017:/root/coco2017 \
        --workdir=/root/mobilenetv2 \
        pocket-mobilenet-monolithic \
        python3 app.monolithic.py

    running_time=$(util_get_running_time mobilenetv2-monolithic-0000)
    echo $running_time > "${rusage_logging_dir}"/mobilenetv2-monolithic-0000.latency

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}

        docker \
            run \
                -d \
                --name ${container_name} \
                --cpus=1 \
                --memory=512mb \
                --volume=$(pwd)/data:/data \
                --volume=$(pwd):/root/mobilenetv2 \
                --volume=/home/cc/COCO/val2017:/root/coco2017 \
                --workdir=/root/mobilenetv2 \
                pocket-mobilenet-monolithic \
                python3 app.monolithic.py
        sleep $(generate_rand_num 3)
    done

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}

        docker wait "${container_name}"
        running_time=$(util_get_running_time "${container_name}")
        echo $running_time > "${rusage_logging_dir}"/"${container_name}".latency
        echo $running_time
    done

    local folder=$(realpath data/${TIMESTAMP}-${numinstances}-graph-monolithic)
    mkdir -p $folder
    for i in $(seq 0 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}
        docker logs $container_name 2>&1 | grep "graph_construction_time" > $folder/$container_name.graph
    done

    folder=$(realpath data/${TIMESTAMP}-${numinstances}-inf-monolithic)
    mkdir -p $folder
    for i in $(seq 0 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}
        docker logs $container_name 2>&1 | grep "inference_time" > $folder/$container_name.inf
    done

    # For debugging
    docker logs -f mobilenetv2-monolithic-$(printf "%04d" $numinstances)
}

function measure_rusage_monolithic() {
    local numinstances=$1
    local container_list=()
    local rusage_logging_dir=$(realpath data/${TIMESTAMP}-${numinstances}-rusage-monolithic)
    local rusage_logging_file=tmp-service.log

    mkdir -p ${rusage_logging_dir}
    init


    # 512mb, oom
    # 512 + 256 = 768mb, oom
    # 1024mb, ok
    # 1024 + 256 = 1280mb
    # 1024 + 512 = 1536mb
    # 1024 + 1024 = 2048mb
    docker \
        run \
            -di \
            --name mobilenetv2-monolithic-0000 \
            --cpus=1 \
            --memory=512mb \
            --volume=$(pwd)/data:/data \
            --volume=$(pwd):/root/mobilenetv2 \
            --volume=/home/cc/COCO/val2017:/root/coco2017 \
            --workdir=/root/mobilenetv2 \
            pocket-mobilenet-monolithic \
            bash

    docker \
        exec \
            mobilenetv2-monolithic-0000 \
            python3 app.monolithic.py

    ../scripts/pocket/pocket \
        rusage \
        measure mobilenetv2-monolithic-0000 --dir ${rusage_logging_dir} 



    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}

        docker \
            run \
                -di \
                --name ${container_name} \
                --cpus=1 \
                --memory=512mb \
                --volume=$(pwd)/data:/data \
                --volume=$(pwd):/root/mobilenetv2 \
                --volume=/home/cc/COCO/val2017:/root/coco2017 \
                --workdir=/root/mobilenetv2 \
                pocket-mobilenet-monolithic \
                bash
    done

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}

        docker \
            exec \
                ${container_name} \
                python3 app.monolithic.py
        sleep $(generate_rand_num 3)
    done

    wait

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}

        ../scripts/pocket/pocket \
            rusage \
            measure ${container_name} --dir ${rusage_logging_dir} 
    done

    # For debugging
    # docker logs -f yolo-monolithic-$(printf "%04d" $numinstances)
}


function measure_perf_monolithic() {
    local numinstances=$1
    local container_list=()
    local rusage_logging_dir=$(realpath data/${TIMESTAMP}-${numinstances}-perf-monolithic)
    local rusage_logging_file=tmp-service.log

    mkdir -p ${rusage_logging_dir}
    init


    # 512mb, oom
    # 512 + 256 = 768mb, oom
    # 1024mb, ok
    # 1024 + 256 = 1280mb
    # 1024 + 512 = 1536mb
    # 1024 + 1024 = 2048mb
    docker \
        run \
            -di \
            --name mobilenetv2-monolithic-0000 \
            --cpus=1 \
            --memory=512mb \
            --volume=$(pwd)/data:/data \
            --volume=$(pwd):/root/mobilenetv2 \
            --volume=/home/cc/COCO/val2017:/root/coco2017 \
            --cap-add SYS_ADMIN \
            --cap-add IPC_LOCK \
            --workdir=/root/mobilenetv2 \
            pocket-mobilenet-monolithic-perf \
            perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -o /data/$TIMESTAMP-${numinstances}-perf-monolithic/mobilenetv2-monolithic-0000.perf.log python3 app.monolithic.py

    docker \
        wait \
            mobilenetv2-monolithic-0000

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}

        docker \
            run \
                -di \
                --name ${container_name} \
                --cpus=1 \
                --memory=512mb \
                --volume=$(pwd)/data:/data \
                --volume=$(pwd):/root/mobilenetv2 \
                --volume=/home/cc/COCO/val2017:/root/coco2017 \
                --cap-add SYS_ADMIN \
                --cap-add IPC_LOCK \
                --workdir=/root/mobilenetv2 \
                pocket-mobilenet-monolithic-perf \
                perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -o /data/$TIMESTAMP-${numinstances}-perf-monolithic/${container_name}.perf.log python3 app.monolithic.py
        sleep $(generate_rand_num 3)
    done

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=mobilenetv2-monolithic-${index}

        docker \
            wait \
                ${container_name}
    done

    # For debugging
    # docker logs -f yolo-monolithic-$(printf "%04d" $numinstances)
}



function measure_rusage() {
    local numinstances=$1
    local container_list=()
    local rusage_logging_dir=$(realpath data/${TIMESTAMP}-${numinstances}-rusage)
    local rusage_logging_file=tmp-service.log

    local server_container_name=pocket-server-001
    local server_image=pocket-mobilenet-server

    mkdir -p ${rusage_logging_dir}
    init
    run_server_basic $server_container_name $SERVER_IP $server_image

    ### rusage measure needs 'd' flag
    ../scripts/pocket/pocket \
        run \
            --rusage $rusage_logging_dir \
            -d \
            -b pocket-mobilenet-application \
            -t pocket-client-0000 \
            -s ${server_container_name} \
            --memory=512mb \
            --cpus=1 \
            --volume=$(pwd)/data:/data \
            --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
            --volume=$(pwd)/../tfrpc/client:/root/tfrpc/client \
            --volume=$(pwd):/root/mobilenetv2 \
            --volume=/home/cc/COCO/val2017:/root/coco2017 \
            --env RSRC_REALLOC_RATIO=${RSRC_RATIO} \
            --env CONTAINER_ID=pocket-client-0000 \
            --workdir='/root/mobilenetv2' \
            -- python3 app.pocket.py &

    ../scripts/pocket/pocket \
        wait \
        pocket-client-0000

    sleep 5

    sudo ../scripts/pocket/pocket \
        rusage \
        init ${server_container_name} --dir ${rusage_logging_dir} 

    ### Firing multiple instances with rusage flag requires & at the end.
    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}

        ../scripts/pocket/pocket \
            run \
                --rusage $rusage_logging_dir \
                -d \
                -b pocket-mobilenet-application \
                -t ${container_name} \
                -s ${server_container_name} \
                --memory=512mb \
                --cpus=1 \
                --volume=$(pwd)/data:/data \
                --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
                --volume=$(pwd)/../tfrpc/client:/root/tfrpc/client \
                --volume=$(pwd):/root/mobilenetv2 \
                --volume=/home/cc/COCO/val2017:/root/coco2017 \
                --env RSRC_REALLOC_RATIO=${RSRC_RATIO} \
                --env CONTAINER_ID=${container_name} \
                --workdir='/root/mobilenetv2' \
                -- python3 app.pocket.py &
        sleep $(generate_rand_num 3)
    done

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}

        # docker wait "${container_name}"
        ../scripts/pocket/pocket \
            wait \
                ${container_name}
    done

    ../scripts/pocket/pocket \
        rusage \
        measure ${server_container_name} --dir ${rusage_logging_dir} 
}

function measure_cprofile() {
    local numinstances=$1
    local container_list=()
    local rusage_logging_dir=$(realpath data/${TIMESTAMP}-${numinstances}-cprofile)
    local rusage_logging_file=tmp-service.log

    local server_container_name=pocket-server-001
    local server_image=pocket-mobilenet-server

    mkdir -p ${rusage_logging_dir}
    init
    run_server_cProfile $server_container_name $SERVER_IP $server_image $TIMESTAMP $numinstances

    ../scripts/pocket/pocket \
        run \
            --cprofile $rusage_logging_dir \
            -d \
            -b pocket-mobilenet-application \
            -t pocket-client-0000 \
            -s ${server_container_name} \
            --memory=512mb \
            --cpus=1 \
            --volume=$(pwd)/data:/data \
            --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
            --volume=$(pwd)/../tfrpc/client:/root/tfrpc/client \
            --volume=$(pwd):/root/mobilenetv2 \
            --volume=/home/cc/COCO/val2017:/root/coco2017 \
            --env RSRC_REALLOC_RATIO=${RSRC_RATIO} \
            --env CONTAINER_ID=pocket-client-0000 \
            --workdir='/root/mobilenetv2' \
            -- python3.6 -m cProfile -o /data/${TIMESTAMP}-${numinstances}-cprofile/pocket-client-0000.cprofile app.pocket.py

    ../scripts/pocket/pocket \
        wait \
        pocket-client-0000

    sleep 5

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}

        ../scripts/pocket/pocket \
            run \
                --cprofile $rusage_logging_dir \
                -d \
                -b pocket-mobilenet-application \
                -t ${container_name} \
                -s ${server_container_name} \
                --memory=512mb \
                --cpus=1 \
                --volume=$(pwd)/data:/data \
                --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
                --volume=$(pwd)/../tfrpc/client:/root/tfrpc/client \
                --volume=$(pwd):/root/mobilenetv2 \
                --volume=/home/cc/COCO/val2017:/root/coco2017 \
                --env RSRC_REALLOC_RATIO=${RSRC_RATIO} \
                --env CONTAINER_ID=${container_name} \
                --workdir='/root/mobilenetv2' \
                -- python3.6 -m cProfile -o /data/${TIMESTAMP}-${numinstances}-cprofile/${container_name}.cprofile app.pocket.py
        sleep $(generate_rand_num 3)
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}

        ../scripts/pocket/pocket \
            wait \
                ${container_name}
    done

    ../scripts/pocket/pocket \
        service \
            kill ${server_container_name} \

    sleep 3

    for filename in data/$TIMESTAMP-${numinstances}-cprofile/* ; do
        echo $filename
        if [[ "$filename" == *.cprofile ]]; then
            ../scripts/pocket/parseprof -f "$filename"
        fi
    done

    # For debugging
    docker logs -f pocket-client-$(printf "%04d" $numinstances)
}

function measure_perf() {
    local numinstances=$1
    local container_list=()
    local rusage_logging_dir=$(realpath data/${TIMESTAMP}-${numinstances}-perf)
    local rusage_logging_file=tmp-service.log

    local server_container_name=pocket-server-001
    local server_image=pocket-mobilenet-server

    mkdir -p ${rusage_logging_dir}
    init
    sudo bash -c "echo 0 > /proc/sys/kernel/nmi_watchdog"

    # sudo python unix_multi_server.py &
    run_server_perf $server_container_name $SERVER_IP $server_image

    ../scripts/pocket/pocket \
        run \
            --perf $rusage_logging_dir \
            -d \
            -b pocket-mobilenet-perf-application \
            -t pocket-client-0000 \
            -s ${server_container_name} \
            --memory=512mb \
            --cpus=1 \
            --volume=$(pwd)/data:/data \
            --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
            --volume=$(pwd)/../tfrpc/client:/root/tfrpc/client \
            --volume=$(pwd):/root/mobilenetv2 \
            --volume=/home/cc/COCO/val2017:/root/coco2017 \
            --env RSRC_REALLOC_RATIO=${RSRC_RATIO} \
            --env CONTAINER_ID=pocket-client-0000 \
            --workdir='/root/mobilenetv2' \
            -- perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -o /data/$TIMESTAMP-${numinstances}-perf/pocket-client-0000.perf.log python3.6 app.pocket.py

    sleep 5


    ../scripts/pocket/pocket \
        wait \
        pocket-client-0000

    sleep 5

    local perf_record_pid=$(sudo ../scripts/pocket/pocket \
        service \
        perf ${server_container_name} --dir ${rusage_logging_dir} --counters cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses)

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}

        ../scripts/pocket/pocket \
            run \
                -d \
                --perf $rusage_logging_dir \
                -b pocket-mobilenet-perf-application \
                -t ${container_name} \
                -s ${server_container_name} \
                --memory=512mb \
                --cpus=1 \
                --volume=$(pwd)/data:/data \
                --volume $(pwd)/../scripts/pocket/tmp/pocketd.sock:/tmp/pocketd.sock \
                --volume=$(pwd)/../tfrpc/client:/root/tfrpc/client \
                --volume=$(pwd):/root/mobilenetv2 \
                --volume=/home/cc/COCO/val2017:/root/coco2017 \
                --env RSRC_REALLOC_RATIO=${RSRC_RATIO} \
                --env CONTAINER_ID=${container_name} \
                --workdir='/root/mobilenetv2' \
                -- perf stat -e cpu-cycles,page-faults,minor-faults,major-faults,cache-misses,LLC-load-misses,LLC-store-misses,dTLB-load-misses,iTLB-load-misses -o /data/$TIMESTAMP-${numinstances}-perf/$container_name.perf.log python3.6 app.pocket.py
        sleep $(generate_rand_num 3)
    done

    sudo bash -c "echo 1 > /proc/sys/kernel/nmi_watchdog"

    for i in $(seq 1 $numinstances); do
        local index=$(printf "%04d" $i)
        local container_name=pocket-client-${index}

        ../scripts/pocket/pocket \
            wait \
                ${container_name}
    done
    sudo kill -s INT $perf_record_pid

    ../scripts/pocket/pocket \
        service \
            kill ${server_container_name} \

    sleep 3

    # For debugging
    docker logs ${server_container_name}
    docker logs -f pocket-client-$(printf "%04d" $numinstances)
}

parse_arg ${@:2}
COMMAND=$1

case $COMMAND in
    build)
        build_docker_files
        ;;

    'run-mon')
        run_monolithic $NUMINSTANCES
        ;;
    'run-poc')
        run_pocket $NUMINSTANCES
        ;;
    'latency-mon')
        measure_latency_monolithic $NUMINSTANCES
        ;;
    'rusage-mon')
        measure_rusage_monolithic $NUMINSTANCES
        ;;
    'perf-mon')
        measure_perf_monolithic $NUMINSTANCES
        ;;
    'latency')
        measure_latency $NUMINSTANCES
        ;;
    'rusage')
        measure_rusage $NUMINSTANCES
        ;;
    'cprofile')
        measure_cprofile $NUMINSTANCES
        ;;
    'perf')
        measure_perf $NUMINSTANCES
        ;;
    'help'|*)
        help
        ;;

esac
#!/bin/bash

BASEDIR=$(dirname $0)
cd $BASEDIR
NUMINSTANCES=1
TIMESTAMP=$(date +%Y%m%d-%H:%M:%S)
NETWORK=tf-grpc-exp

EXP_ROOT="${HOME}/settings/tf-slim/lightweight/pjt/grpc"


GET_CONTAINER_PID='{{.State.Pid}}'
GET_CONTAINER_ID='{{.Id}}'
GET_CONTAINER_CREATED='{{.Created}}'
GET_CONTAINER_STARTED='{{.State.StartedAt}}'
GET_CONTAINER_FINISHED='{{.State.FinishedAt}}'
GET_CONTAINER_IPADDRESS='{{.NetworkSettings.IPAddress}}'

source internal_functions.sh

function init() {
    docker rm -f $(docker ps -a | grep "grpc_server\|grpc_app_\|grpc_exp_server\|grpc_exp_app_\|" | awk '{print $1}') > /dev/null 2&>1
    docker network rm $NETWORK
    docker network create --driver=bridge $NETWORK
}

function health_check() {
    init
    # Run a server
    _run_server grpc_exp_server grpc_exp_server $NETWORK 5

    # Run a client with hello
    _run_client grpc_exp_client grpc_exp_app_00 $NETWORK
}

function build_image() {
    docker rmi -f grpc_exp_server grpc_exp_client

    docker image build --no-cache -t grpc_exp_client -f dockerfiles/Dockerfile.idapp .
    docker image build --no-cache -t grpc_exp_server -f dockerfiles/Dockerfile.idser .
}

function help() {
    echo Usage: ./exp_script.sh COMMAND [OPTIONS]
    echo Supported Commands:
    echo -e '\thealth, help, build'
}

COMMAND=$([[ $# == 0 ]] && echo help || echo $1)
case $COMMAND in
    build)
        build_image
        ;;
    health)
        health_check
        ;;
    *|help)
        help
        ;;
esac
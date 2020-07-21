#!/bin/bash

# SERVER_ADDR='localhost' python3.6 detect.py --image data/meme.jpg

container_name=grpc_client_test
mkdir -p data
docker rm -f $container_name > /dev/null 2>&1
docker create \
    -it \
    --volume=$(pwd)/data:/data \
    --network=tf-grpc \
    --name=$container_name \
    --workdir='/root/yolov3-tf2' \
    --env SERVER_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' grpc_server) \
    grpc_client_id_only \
    python detect.py --image data/meme.jpg

docker start $container_name 
docker logs --follow grpc_client_test
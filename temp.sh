#!/bin/bash

SERVER_ADDR='localhost' python3.6 detect.py --image data/meme.jpg
# SERVER_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' grpc_server) python3.6 detect.py --image data/meme.jpg
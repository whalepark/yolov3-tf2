#!/bin/bash

rm -f ../client/{yolo_pb2_grpc.py,yolo_pb2.py}
rm -f ../server/{yolo_pb2_grpc.py,yolo_pb2.py}

python3 -m grpc_tools.protoc -I. --python_out=../server --grpc_python_out=../server yolo.proto
python3 -m grpc_tools.protoc -I. --python_out=../client --grpc_python_out=../client yolo.proto

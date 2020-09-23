import sys, os
sys.path.append('proto')
import logging
import time
import argparse
import subprocess

import grpc
import exp_pb2, exp_pb2_grpc
from concurrent import futures


class Experiments():
    @staticmethod
    def Echo(stub, data):
        request = exp_pb2.EchoRequest()
        response: exp_pb2.EchoResponse
        
        request.data = data
        response = stub.Echo(request)

        return response.data

    @staticmethod
    def SendFilePath(stub, container_id, path):
        request = exp_pb2.SendFilePathRequest()
        response: exp_pb2.SendFilePathResponse
        
        request.container_id = container_id
        request.path = path

        start = time.time()
        response = stub.SendFilePath(request)
        end = time.time()
        logging.info(f'rtt(transmit_id)={end-start}')

        return response

    @staticmethod
    def SendFileBinary(stub, bin):
        request = exp_pb2.SendFileBinaryRequest()
        response: exp_pb2.SendFileBinaryResponse
        
        request.bin = bin
        start = time.time()
        response = stub.SendFileBinary(request)
        end = time.time()
        logging.info(f'rtt(transmit_bin)={end-start}')

        return response

    @staticmethod
    def ServerIOLatency(stub, container_id, path):
        request = exp_pb2.ServerIOLatencyRequest()
        response: exp_pb2.ServerIOLatencyResponse
        
        request.container_id = container_id
        request.path = path

        response = stub.ServerIOLatency(request)
        logging.info(f'latency(file_io_server)={response.log}')

        return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', help='file path', type=str, required=True)
    args = parser.parse_args()
    file = os.path.abspath(args.file)

    server_addr = os.environ.get('SERVER_ADDR')
    # server_addr = 'localhost'
    container_id = subprocess.check_output('cat /proc/self/cgroup | grep cpuset | cut -d/ -f3 | head -1', shell=True, encoding='utf-8').strip()


    logging.info(server_addr)
    channel = grpc.insecure_channel(f'{server_addr}:1991', \
        options=(('grpc.max_send_message_length', 100 * 1024 * 1024), \
        ('grpc.max_receive_message_length', 100 * 1024 * 1024), \
        ('grpc.max_message_length', 100 * 1024 * 1024),
        ('grpc.enable_http_proxy', 0)) \
    )
    stub = exp_pb2_grpc.ExperimentServiceStub(channel)

    returned_data = Experiments.Echo(stub, 'Hello Misun!')
    logging.info(f'returned_data={returned_data}')

    Experiments.SendFilePath(stub, container_id, file)

    start = time.time()
    read_image = open(file, 'rb').read()
    end = time.time()
    logging.info(f'latency(file_io_client)={end-start}')

    Experiments.SendFileBinary(stub, read_image)

    Experiments.ServerIOLatency(stub, container_id, file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, \
                        format='[%(asctime)s|CLIENT] %(message)s')
    main()
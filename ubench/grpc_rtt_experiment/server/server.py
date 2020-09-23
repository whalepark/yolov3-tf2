import sys
sys.path.append('proto')
import logging
import subprocess

import grpc
import exp_pb2, exp_pb2_grpc
from concurrent import futures
import time

def debug_ls(dir: str):
    output = subprocess.check_output(f'ls -al {dir}', shell=True, encoding='utf-8').strip()
    logging.debug(output)


class ExperimentSet(exp_pb2_grpc.ExperimentServiceServicer):
    def Echo(self, request, context):
        print(f'Echo')
        response = exp_pb2.EchoResponse()
        
        response.data = request.data

        return response

    def SendFilePath(self, request, context):
        print(f'SendFilePath')
        response = exp_pb2.SendFilePathResponse()

        return response

    def SendFileBinary(self, request, context):
        print(f'SendFileBinary')
        response = exp_pb2.SendFileBinaryResponse()

        return response

    def ServerIOLatency(self, request, context):
        print(f'ServerIOLatency')
        response = exp_pb2.ServerIOLatencyResponse()
        
        path = request.path
        container_id = request.container_id
        prefix = '/layers/' + subprocess.check_output('docker inspect -f {{.GraphDriver.Data.MergedDir}} ' + container_id, shell=True).decode('utf-8').strip().strip('/').split('/')[4] + '/merged'
        
        # debug_ls('/')
        # debug_ls('/layers')
        # debug_ls(prefix)

        full_path = prefix + path
        start = time.time()
        read_img = open(full_path, 'rb').read()
        end = time.time()
        response.log = str(end-start)

        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=47),
                         options=[('grpc.so_reuseport', 1),
                                  ('grpc.max_send_message_length', -1),
                                  ('grpc.max_receive_message_length', -1)])
    exp_pb2_grpc.add_ExperimentServiceServicer_to_server(ExperimentSet(), server)
    server.add_insecure_port('[::]:1991')
    server.start()
    logging.info('Server started!')
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, \
                        format='[%(asctime)s|SERVER] %(message)s')
    serve()
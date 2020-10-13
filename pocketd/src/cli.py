import logging # debug, info, warning, error, critical...
import time
import os
import json
import socket

import argparse
import sys

daemon_root = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
logdir = os.path.join(daemon_root, 'logs')
assetdir = os.path.join(daemon_root, 'assets')
starttime = time.time()
file_handler = None
logger = None
pocket_socket_path = os.path.join(assetdir, 'pocket.sock')

def init_logger(debug_level=logging.INFO,
                log_file=None):
    global logger, file_handler

    if log_file is None:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(starttime))
        log_file=f'{logdir}/cli-{timestamp}.log'

    logger = logging.getLogger('pocketcli_logger')
    logger.setLevel(debug_level)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    
    # stream_handler = logging.StreamHandler()
    # logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def parse_arg():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands', help='additional help')
    # parser_start = subparsers.add_parser(start)
    # parser_stop = subparsers.add_stop(stop)

def remove():
    print('remove')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0) as sock:
        sock.connect(pocket_socket_path)
        data = {'command': 'remove'}
        sock.sendall(json.dumps(data).encode('utf8')) #, sort_keys=True)
        response = str(sock.recv(1024), 'ascii')
        print(response)


def main():
    init_logger(logging.DEBUG)
    # parse_arg()
    command = sys.argv[1]
    if command == 'create':
        logger.debug('pocket create request')
    elif command == 'rm':
        logger.debug('pocekt rm request')
        remove()
    elif command == 'ls':
        logger.debug('pocekt ls request')
    elif command == 'run':
        logger.debug('pocekt run request')
    elif command == 'start':
        logger.debug('pocekt start request')
    elif command == 'inspect':
        logger.debug('pocekt inspect request')
    elif command == 'profile':
        logger.debug('pocekt profile request')
    elif command == 'service':
        logger.debug('pocekt service')
    else:
        print('unknown argument')

if __name__ == '__main__':
    main()


# python rest api: https://www.nylas.com/blog/use-python-requests-module-rest-apis/
# https://my-devblog.tistory.com/27
# python rest api serving: https://rekt77.tistory.com/103

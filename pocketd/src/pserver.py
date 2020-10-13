import logging
import os
import time
import atexit

from pocket.pocket import PocketServer
from pocket.redis import RedisManager

logger = None
daemon_root = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
logdir = os.path.join(daemon_root, 'logs')
assetdir = os.path.join(daemon_root, 'assets')
starttime = time.time()
pocket_socket_path = os.path.join(assetdir, 'pocket.sock')
print(pocket_socket_path)
CONCURRENT_CONNECTIONS = 1


def init_logger(debug_level=logging.INFO,
                log_file=None):
    global logger

    if log_file is None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(starttime))
        log_file=f'{logdir}/server-{timestamp}.log'

    logger = logging.getLogger('pocketd')
    logger.setLevel(debug_level)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    
    # stream_handler = logging.StreamHandler()
    # logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def init_redis():
    # create redis container
    # its network ns should be a part of pocket network ns.

    pass


def finalize():
    if os.path.exists(pocket_socket_path):
        os.unlink(pocket_socket_path)
    # destruct redis
    logger.info('Being terminated, bye-bye!')

def initialize():
    init_logger(logging.DEBUG)
    init_redis()
    atexit.register(finalize)
    logger.info('pocket initialized')



def main():
    initialize()

    # Todo
    # service_mgr = pocket.service.Service()
    # redis_mgr = RedisManager()
    # app_mgr = pocket.application.Application()

    pserver = PocketServer(pocket_socket_path, logger)

    pserver.create_and_bind_sockets()
    pserver.run_server()


if __name__ == '__main__':
    main()
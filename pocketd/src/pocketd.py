from daemonize import Daemonize # https://github.com/thesharp/daemonize/blob/master/daemonize.py
import logging # debug, info, warning, error, critical...
import time
import os
import signal

import pserver
import argparse
import sys, fcntl

daemon_root = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
logdir = os.path.join(daemon_root, 'logs')
assetdir = os.path.join(daemon_root, 'assets')
starttime = time.time()
file_handler = None
logger = None
pid_lock = os.path.join(assetdir, 'pocket.pid')

def init_logger(debug_level=logging.INFO,
                log_file=None):
    global logger, file_handler

    if log_file is None:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(starttime))
        log_file=f'{logdir}/daemon-{timestamp}.log'

    logger = logging.getLogger('daemon_logger')
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

def main():
    init_logger(logging.DEBUG)
    # parse_arg()

    if sys.argv[1] == 'start':
        logger.info('daemon start request')
        if os.path.exists(pid_lock):
            print('daemon running already. exit.')
            logger.warning('daemon running already. exit.')
            exit(1)
        daemon = Daemonize(app='pocketd', pid=pid_lock, action=pserver.main, logger=logger) #, foreground=True, keep_fds=[0,1,2])
        daemon.start()
    elif sys.argv[1] == 'stop':
        logger.info('daemon stop request')
        if os.path.exists(pid_lock):
            try:
                lockfile = open(pid_lock, 'r')
                pid = int(lockfile.read().strip())
                print(pid)
                logger.info(f'kill the daemon (pid={pid})')
                os.kill(pid, signal.SIGTERM)
            except:
                logger.warning('some exception')
        else:
            print('No daemon running currently.')
    else:
        print('unknown argument')

if __name__ == '__main__':
    main()


    

### Reference
# https://gist.github.com/tzuryby/961228
# https://oddpoet.net/blog/2013/09/24/python-daemon/
# https://dpbl.wordpress.com/2017/02/12/a-tutorial-on-python-daemon/
# http://blog.cloudsys.co.kr/python-daemon-example-install/

### Intercept python module function
# https://stackoverflow.com/questions/35758323/hook-python-module-function

import os
import sys
import socket
import logging
import signal
import multiprocessing

SERVER_SOCKET_PATH = './sockets/server.sock'

SERVER_SOCKET: socket.socket

CONCURRENT_CONNECTIONS = 1

MULTIPROC = True

def remove_remaining_sockets():
    pwd = os.getcwd()
    if not os.path.exists(f'{pwd}/sockets'):
        os.makedirs(f'{pwd}/sockets', exist_ok=True)

    if os.path.exists(f'{pwd}/{SERVER_SOCKET_PATH}'):
        os.unlink(SERVER_SOCKET_PATH)

def create_and_bind_sockets():
    global SERVER_SOCKET

    SERVER_SOCKET = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    SERVER_SOCKET.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    SERVER_SOCKET.bind(SERVER_SOCKET_PATH)

def run_server():
    SERVER_SOCKET.listen(CONCURRENT_CONNECTIONS)
    while True:
        conn, _addr = SERVER_SOCKET.accept()
        if MULTIPROC: 
            process = multiprocessing.Process(target=handle_client, args=(conn, _addr))
            process.daemon = True
            process.start()
        else:
            data_received = conn.recv(1024)
            do_something(data_received)
            data_to_send = data_received
            conn.send(data_to_send)
            conn.close()

def do_something(data_received):
    print(data_received)
    pid = int(data_received)
    import subprocess
    output = subprocess.check_output(f'sudo perf stat -e cycles -p {pid}', shell=True, encoding='utf-8')
    print(output)

def handle_client(conn, addr):
    while True:
        data_received = conn.recv(1024)
        do_something(data_received)
        data_to_send = data_received
        conn.send(data_to_send)
        break
    conn.close()

def finalize(signum, frame):
    print('finalizing workers...')
    for process in multiprocessing.active_children():
        # logging.info("Shutting down process %r", process)
        process.terminate()
        process.join()
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, finalize)

    remove_remaining_sockets()
    create_and_bind_sockets()
    run_server()
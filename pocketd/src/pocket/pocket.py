import os
import socket, threading
import logging
from threading import Thread

class PocketServer:
    def __init__(self, socket_path, logger, multi_thread=True):
        self.socket_path = socket_path
        self.socket = None
        self.logger = logger
        self.multi_thread = multi_thread

    def init_socket(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def create_and_bind_sockets(self):
        self.init_socket()
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.socket_path)

    def do_something(self, data):
        print(data)

    def handle_client(self, conn, addr):
        while True:
            data_received = conn.recv(1024).decode('utf-8')
            data_to_send = data_received.encode('utf-8')
            # actual handle
            self.do_something(data_to_send)

            # response
            conn.send(data_to_send)
            break
        conn.close()

    def run_server(self):
        self.socket.listen()
        while True:
            conn, _addr = self.socket.accept()
            if self.multi_thread: 
                t = Thread(target=self.handle_client, args=(conn, _addr))
                t.daemon = True
                t.start()
            else:
                handle_client(conn, _addr)

# python rest api: https://www.nylas.com/blog/use-python-requests-module-rest-apis/
# https://my-devblog.tistory.com/27
# python rest api serving: https://rekt77.tistory.com/103
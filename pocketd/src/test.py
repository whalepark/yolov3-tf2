import os
import socket
import time
import sys

daemon_root = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
logdir = os.path.join(daemon_root, 'logs')
assetdir = os.path.join(daemon_root, 'assets')
starttime = time.time()
pocket_socket_path = os.path.join(assetdir, 'pocket.sock')

data = " ".join(sys.argv[1:])

print(pocket_socket_path)

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0) as sock:
    sock.connect(pocket_socket_path)
    sock.sendall(bytes(data, 'ascii'))
    response = str(sock.recv(1024), 'ascii')
    # s.sendall(b'GET / HTTP/1.1\n')
    print(response)


print("Sent:     {}".format(data))
print("Received: {}".format(response))
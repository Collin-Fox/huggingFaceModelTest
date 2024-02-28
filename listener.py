import socket,time

import main

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket.connect(('localhost', 5050))

while True:
    time.sleep(5)
    data = client_socket.recv(1024)
    message = data.decode("utf-8")
    test = main.get_prompt_inp(message)
    print(message)
    print(test)
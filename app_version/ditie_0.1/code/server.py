#coding: utf-8
import json
import socket
import sys
def socket_service_data():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        s.bind(('localhost', 7775))  # 绑定要监听的端口
        s.listen(5)  # 开始监听 表示可以使用五个链接排队
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print("Wait for Connection..................")
    while True:
        sock, addr = s.accept()
        buf = sock.recv(1024*30)  #接收数据
        # buf = buf.decode()  #解码
        buf = json.loads(buf.decode())
        print(type(buf))
        print(buf)
        dict1 = {"ItemName": "顶板梁混凝土"}
        sock.send(json.dumps(dict1, ensure_ascii=False).encode())
    sock.close()
socket_service_data()

#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = "marie"
__baseAuthor__ = "omori"
__version__ = "1.0.0"

import argparse
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from estimation import discriminator


class MyHandler(BaseHTTPRequestHandler):
    """
    Received the request as json, send the response as json
    please you edit the your processing
    """
    global realsense
    def do_POST(self):
        try:
            content_len = int(self.headers.get('content-length'))
            requestBody = self.rfile.read(content_len).decode('utf-8')

            print(requestBody)

            response = realsense.detect()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            responseBody = json.dumps(response)

            self.wfile.write(responseBody.encode('utf-8'))
        except Exception as e:
            print("An error occured")
            print("The information of error is as following")
            print(type(e))
            print(e.args)
            print(e)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            responseBody = json.dumps(response)

            self.wfile.write(responseBody.encode('utf-8'))


def importargs():
    parser = argparse.ArgumentParser("This is the simple server")

    parser.add_argument('--host', '-H', required=False, default='localhost')
    parser.add_argument('--port', '-P', required=False, type=int, default=8080)

    args = parser.parse_args()

    return args.host, args.port


def run(server_class=HTTPServer, handler_class=MyHandler, server_name='localhost', port=8080):

    server = server_class((server_name, port), handler_class)
    server.serve_forever()

def main():
    global realsense
    host, port = importargs()
    realsense = discriminator.Realsense(0.5)

    run(server_name=host, port=port)



if __name__ == '__main__':
    main()

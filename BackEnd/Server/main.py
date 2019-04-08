#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.wsgi
import tornado.websocket
import json
import sys
from storage import Storage


storage_file = "storage.db"
port = 8080

class WebSocketAPI(tornado.websocket.WebSocketHandler):
    storage_handler = None

    def check_origin(self, origin):
        return True

    def on_message(self, msg):
        # Parse
        data = json.loads(msg)

        # Save data
        WebSocketAPI.storage_handler.write(data["et"], data["t"], data["v"])


if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) > 1:
        storage_file = sys.argv[1]

    # Create data for storage
    storage = Storage(storage_file)
    storage.create()

    WebSocketAPI.storage_handler = storage

    try:
        # Create server
        app = tornado.web.Application([('/api', WebSocketAPI),])
        server = tornado.httpserver.HTTPServer(app)
        
        # Run server
        server.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    except Exception as ex:
        print(str(ex))
    finally:
        storage.close()

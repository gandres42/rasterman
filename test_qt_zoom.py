import os
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"
import webview
import time
import sys

def on_start():
    time.sleep(1)
    sys.exit(0)

window = webview.create_window('Test', html='<body><h1>Test</h1></body>')
webview.start(on_start)

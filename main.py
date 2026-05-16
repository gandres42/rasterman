import webview
import time
window = webview.create_window('Rasterman', 'index.html', frameless=True, easy_drag=False, width=960, height=520, resizable=True, min_size=(960, 510))
webview.start()
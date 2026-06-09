import os
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

import importlib
import webview
import time
import base64
import os

import cv2
import numpy as np
import rembg

class Api:
    def __init__(self):
        self.last_result = None
        self.current_image_path = None
        self.current_image_data = None
        self.bg_cache = {}

    def minimize(self):
        window.minimize()

    def maximize(self):
        window.maximize()

    def close(self):
        window.destroy()

    def select_image(self):
        file_types = ('Image Files (*.bmp;*.gif;*.jpg;*.jpeg;*.png;*.webp)', 'All files (*.*)')
        result = window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types)
        if result and len(result) > 0:
            self.current_image_path = result[0]
            try:
                img = cv2.imread(self.current_image_path)
                if img is None:
                    return None

                h, w = img.shape[:2]
                side = min(h, w)
                y0 = (h - side) // 2
                x0 = (w - side) // 2
                img = img[y0:y0 + side, x0:x0 + side]

                self.current_image_data = img
                self.bg_cache.clear()
                
                _, buffer = cv2.imencode('.png', img)
                encoded_str = base64.b64encode(buffer).decode('utf-8')
                return f"data:image/png;base64,{encoded_str}"
            except Exception as e:
                print(f"Failed to read image for UI preview: {e}")
        return None

    def process_image(self, threshold_val, threshold_type_str, remove_bg, rembg_model, downsample_str, result_size):
        if self.current_image_data is None:
            return None
        
        try:
            img = self.current_image_data.copy()
            
            # 1. Remove background / fetch from cache
            if remove_bg == 'true':
                if rembg_model not in self.bg_cache:
                    session = rembg.new_session(rembg_model)
                    self.bg_cache[rembg_model] = rembg.remove(img, session=session)
                img = self.bg_cache[rembg_model].copy()

            # 2. Pixelate based on longest edge
            h, w = img.shape[:2]
            scale = int(result_size) / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            
            # Extract interpolation method for downsampling
            down_interp_attr = downsample_str.split('.')[-1]
            down_interp = getattr(cv2, down_interp_attr, cv2.INTER_AREA)

            # Scale down to logical size
            img_small = cv2.resize(img, (new_w, new_h), interpolation=down_interp)
            # Scale back up with nearest-neighbor for the pixelated display preview
            img_display = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_NEAREST)

            thresh_attr = threshold_type_str.split('.')[-1]
            thresh_type = getattr(cv2, thresh_attr, cv2.THRESH_BINARY)

            def apply_threshold(src):
                if src.shape[2] == 4:
                    gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
                    alpha = src[:, :, 3]
                    _, alpha = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
                    _, thresh = cv2.threshold(gray, int(threshold_val), 255, thresh_type)
                    out = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGRA)
                    out[:, :, 3] = alpha
                else:
                    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                    _, out = cv2.threshold(gray, int(threshold_val), 255, thresh_type)
                return out

            # Save at downsampled resolution; display at original resolution
            self.last_result = apply_threshold(img_small)
            result = apply_threshold(img_display)
            
            # 4. Encode to base64 for pywebview
            _, buffer = cv2.imencode('.png', result)
            encoded_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{encoded_str}"
            
        except Exception as e:
            print(f"Image processing error: {e}")
            return f"Error: {e}"

    def save_image(self):
        if self.last_result is None:
            return
            
        # Create a copy so we don't accidentally modify the active image in the app
        img_to_save = self.last_result.copy()
        
        # Check if the image has an alpha channel (4 channels: BGRA)
        if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 4:
            # Extract the alpha channel and normalize to 0.0 - 1.0
            # Keeping the slice as 3: preserves the 3rd dimension for broadcasting
            alpha = img_to_save[:, :, 3:] / 255.0 
            bgr = img_to_save[:, :, :3]
            
            # Blend the BGR channels with a white background (255)
            img_to_save = (alpha * bgr + (1 - alpha) * 255).astype(np.uint8)
        
        target_path = window.create_file_dialog(
            webview.SAVE_DIALOG, 
            directory='', 
            save_filename='result.jpg'
        )
        
        # Save the processed image
        if target_path and isinstance(target_path, tuple):
            cv2.imwrite(target_path[0], img_to_save)
        elif target_path:
            cv2.imwrite(target_path, img_to_save)

api = Api()
window = webview.create_window('Rasterman', 'index.html', js_api=api, frameless=True, easy_drag=False)
webview.start()
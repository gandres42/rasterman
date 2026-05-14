import cv2
from rembg import remove, new_session
import numpy as np
from typing import Optional

def main():
    print('loading model...')
    session = new_session(model_name="bria-rmbg")

    print('processing image...')
    img: Optional[np.ndarray] = cv2.imread('./lighthouse.jpg')
    if img is None:
        raise FileNotFoundError
    img: np.ndarray = remove(img, session=session)  # ty:ignore[invalid-assignment]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)

    w, h = (8, 8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

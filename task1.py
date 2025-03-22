import cv2
import numpy as np

img = cv2.imread('images\\variant-4.jpeg')
b, g, r = cv2.split(img)
cv2.imshow('b', np.array([list(map(lambda x: [x, np.uint8(0), np.uint8(0)], line)) for line in b]))
cv2.waitKey(0)



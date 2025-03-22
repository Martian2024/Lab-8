import cv2
import numpy as np

# image = cv2.imread('images\\variant-3.jpeg')
# image = cv2.resize(image, list(map(lambda x: x // 2, image.shape[:2]))[::-1])

def get_reference_contours():
    img = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
    _, treshed = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(treshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(sorted(contours, key=len, reverse=True))
    return contours[:3]

def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, treshed = cv2.threshold(gray_frame, 125, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(treshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x: len(x) > 25, contours))
    contours = list(sorted(contours, key=len, reverse=True))


    best_fitting = [min(contours, key=lambda x: cv2.matchShapes(x, i, 1, 0.0)) for i in reference_contours]
    x, y, w, h = cv2.boundingRect(np.concatenate(best_fitting))


    cv2.drawContours(frame, best_fitting, -1, (0, 255, 0), 3)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    return frame

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    reference_contours = get_reference_contours()

    while True:
        ret, frame = cam.read()
        gray_frame = process_frame(frame)
        cv2.imshow('AAA', gray_frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
import cv2
import numpy as np

# image = cv2.imread('images\\variant-3.jpeg')
# image = cv2.resize(image, list(map(lambda x: x // 2, image.shape[:2]))[::-1])

def place_image(frame, image, x, y):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if 0 <= y - image.shape[0] // 2 + i < frame.shape[0] and 0 <= x - image.shape[1] // 2 + j < frame.shape[1]:
                frame[y - image.shape[0] // 2 + i][x - image.shape[1] // 2 + j] = image[i][j]
    return frame

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

    frame = place_image(frame, fly, x + w // 2, y + h // 2)

    cv2.drawContours(frame, best_fitting, -1, (0, 255, 0), 3)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    return frame

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    reference_contours = get_reference_contours()
    fly = cv2.imread('fly64.png')

    while True:
        ret, frame = cam.read()
        frame = process_frame(frame)
        cv2.imshow('AAA', frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
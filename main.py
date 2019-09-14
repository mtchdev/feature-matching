import sys
import cv2
import numpy as np
from matplotlib import pyplot

query = cv2.imread("1.png", 0)
train = cv2.imread("2.png", 0)

def analyze(query, train):
    orb = cv2.ORB_create()

    # find key points
    x1, y1 = orb.detectAndCompute(query, None)
    x2, y2 = orb.detectAndCompute(train, None)

    # measure distances, verify accuracy
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    keyf = sorted(bf.match(y1, y2), key = lambda x: x.distance)
    final = cv2.drawMatches(query, x1, train, x2, keyf[:10], None, flags=2)

    pyplot.imshow(final)
    pyplot.draw()
    pyplot.pause(0.01)
    
    if not pyplot.get_fignums():
        pyplot.show()

if __name__ == "__main__":
    
    mode = input("Mode (image/video): ")
    if mode == "video":
        video = input("Video (query): ")
        model = input("Image (model): ")
        cap = cv2.VideoCapture(video)
        model = cv2.imread(model, 0)

        while True:
            ret, frame = cap.read()

            analyze(frame, model)

            cv2.waitKey(1000)

        cap.release()

    elif mode == "image":
        img1 = input("Image 1 (query): ")
        img2 = input("Image 2 (model): ")
        query = cv2.imread(img1, 0)
        train = cv2.imread(img2, 0)

        analyze(query, train)
    else:
        print("No valid option selected.")
        sys.exit()


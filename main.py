import cv2
import numpy as np
from matplotlib import pyplot

query = cv2.imread("1.png", 0)
train = cv2.imread("2.png", 0)

orb = cv2.ORB_create()
print(query)
# find key points
x1, y1 = orb.detectAndCompute(query, None)
x2, y2 = orb.detectAndCompute(train, None)

# measure distances, verify accuracy
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

keyf = sorted(bf.match(y1, y2), key = lambda x: x.distance)
final = cv2.drawMatches(query, x1, train, x2, keyf[:10], None, flags=2)

# draw matches
pyplot.imshow(final),pyplot.show()


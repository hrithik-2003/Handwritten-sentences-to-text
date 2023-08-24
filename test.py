import cv2

image = cv2.imread('images\Sentences\sen_1.jpg')
image_copy = image.copy()

cv2.imshow('out', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
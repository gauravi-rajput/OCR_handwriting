import cv2
import pytesseract
import numpy as np

#read the image from folder

# image = cv2.imread(‘Presc7.jpg’)

image = cv2.imread('demo/print.png')
if image is None:
    raise FileNotFoundError("Image not found at 'demo/presc1.png'")
scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
image_resize = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
fromCenter = False
showCrosshair = False
r = cv2.selectROI(image_resize, fromCenter, showCrosshair)
# Crop the image to the selected region
imCrop = image_resize[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
scale_percent = 220 # percent of original size
width = int(imCrop.shape[1] * scale_percent / 100)
height = int(imCrop.shape[0] * scale_percent / 100)
dim = (width, height)
print(dim)
print(imCrop)
# resize the cropped image
resized = cv2.resize(imCrop, dim, interpolation = cv2.INTER_AREA)
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
cv2.imshow('Grey Scale', hsv)
lower = np.array([0, 0, 0])
upper = np.array([200, 200, 200])
# upper = np.array([175, 180, 150]) #typed
mask = cv2.inRange(hsv, lower, upper)
invert = 255 - mask

data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6 --oem 1')
print(data)

cv2.imshow('mask', mask)
cv2.imshow('invert', invert)
cv2.waitKey()
cv2.destroyAllWindows()




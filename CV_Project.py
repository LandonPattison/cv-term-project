import cv2
import numpy as np
import os

trainDirectory = 'ProjData/MiniTest'
saveDirectory = 'ProjData/MiniTestResized'
siftDirectory = 'ProjData/MiniSift'
histDirectory = 'ProjData/MiniHist'

sizes = [(200, 200), (50, 50)]
sift = cv2.SIFT_create()

if not os.path.exists(saveDirectory):
    os.makedirs(saveDirectory)

if not os.path.exists(siftDirectory):
    os.makedirs(siftDirectory)

if not os.path.exists(histDirectory):
    os.makedirs(histDirectory)

# Pre-Processing 
for imgName in os.listdir(trainDirectory):
    imgPath = os.path.join(trainDirectory, imgName)
    img = cv2.imread(imgPath)

    # a. Convert to grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # adjust the brightness if necessary 
    averageB = np.mean(gray)
    if averageB < 0.4:
        gray = cv2.addWeighted(gray, 1.5, np.zeros(gray.shape, dtype=gray.dtype), 0, 0)
    elif averageB > 0.6:
        gray = cv2.addWeighted(gray, 0.5, np.zeros(gray.shape, dtype=gray.dtype), 0, 0)

    # b. Resize the image to TWO different sizes: 200*200 and 50*50 and save them. 
    for size in sizes:
        resized = cv2.resize(gray, size)
        cv2.imwrite(f"{saveDirectory}/{imgName.split('.')[0]}_{size[0]}x{size[1]}.jpg", resized)
        #cv2.imshow('Resized', resized)S
        #cv2.waitKey(0)


        # 3. Extract Histogram features on ALL training images and save the data. 
        hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
        histPath = os.path.join(histDirectory, f"{imgName.split('.')[0]}_{size[0]}x{size[1]}.npy")
        np.save(histPath, hist)

        # 2. Extract SIFT features on ALL training images and save the data.
        keyPoints, descriptor = sift.detectAndCompute(resized, None)
        siftPath = os.path.join(siftDirectory, f"{imgName.split('.')[0]}_{size[0]}x{size[1]}.npy")
        np.save(siftPath, descriptor)



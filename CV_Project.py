import cv2
import numpy as np
import os

trainDirectory = 'C:/Users/maria/Computer Vision/ProjData/MiniTest'
saveDirectory = 'C:/Users/maria/Computer Vision/ProjData/MiniTestResized'
siftDirectory = 'C:/Users/maria/Computer Vision/ProjData/MiniSift'
sizes = [(200, 200), (50, 50)]
sift = cv2.SIFT_create()

# loop through train image directory
for imgName in os.listdir(trainDirectory):
    imgPath = os.path.join(trainDirectory, imgName)
    img = cv2.imread(imgPath)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # average brightness of image and adjusting if necessary
    avgBrightness = np.mean(grayImg)

    if avgBrightness < 0.4:
        grayImg = cv2.addWeighted(grayImg, 1.5, np.zeros(grayImg.shape, dtype=grayImg.dtype), 0, 0)
    elif avgBrightness > 0.6:
        grayImg = cv2.addWeighted(grayImg, 0.5, np.zeros(grayImg.shape, dtype=grayImg.dtype), 0, 0)

    # resizing images and saving them
    for size in sizes:
        resized = cv2.resize(grayImg, size)

        # saving the resized image in a different directory
        #cv2.imwrite(f"{saveDirectory}/{imgName.split('.')[0]}_{size[0]}x{size[1]}.jpg", resized)

        #debugging: outputting the images to check
        cv2.imshow('Resized Image', resized)
        cv2.waitKey(0)


    # Extract SIFT features and save the data
    keyPoints, descriptor = sift.detectAndCompute(grayImg, None)
    siftPath = os.path.join(siftDirectory, f"{imgName.split('.')[0]}.npy")
    #np.save(siftPath, descriptor)

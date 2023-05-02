import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import collections

def preprocess_data(input_directory, save_directory, sift_directory, hist_directory):
    sizes = [(200, 200), (50, 50)]
    sift = cv2.SIFT_create()

    for subdir in os.listdir(input_directory):
        subdir_path = os.path.join(input_directory, subdir)
        if os.path.isdir(subdir_path):
            save_subdir = os.path.join(save_directory, subdir)
            sift_subdir = os.path.join(sift_directory, subdir)
            hist_subdir = os.path.join(hist_directory, subdir)

            os.makedirs(save_subdir, exist_ok=True)
            os.makedirs(sift_subdir, exist_ok=True)
            os.makedirs(hist_subdir, exist_ok=True)

            for img_name in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img_name)
                img = cv2.imread(img_path)

                # a. Convert to grayscale images
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Adjust the brightness if necessary
                averageB = np.mean(gray)
                if averageB < 0.4:
                    gray = cv2.addWeighted(gray, 1.5, np.zeros(gray.shape, dtype=gray.dtype), 0, 0)
                elif averageB > 0.6:
                    gray = cv2.addWeighted(gray, 0.5, np.zeros(gray.shape, dtype=gray.dtype), 0, 0)

                # b. Resize the image to TWO different sizes: 200*200 and 50*50 and save them.
                for size in sizes:
                    resized = cv2.resize(gray, size)
                    cv2.imwrite(f"{save_subdir}/{img_name.split('.')[0]}_{size[0]}x{size[1]}.jpg", resized)

                    # 3. Extract Histogram features on ALL images and save the data.
                    hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
                    hist_path = os.path.join(hist_subdir, f"{img_name.split('.')[0]}_{size[0]}x{size[1]}.npy")
                    np.save(hist_path, hist)

                    # 2. Extract SIFT features on ALL images and save the data.
                    key_points, descriptor = sift.detectAndCompute(resized, None)
                    sift_path = os.path.join(sift_subdir, f"{img_name.split('.')[0]}_{size[0]}x{size[1]}.npy")
                    np.save(sift_path, descriptor)

# Pre-process Train data
train_directory = 'ProjData/Train'
train_save_directory = 'ProjData/TrainResized'
train_sift_directory = 'ProjData/TrainSift'
train_hist_directory = 'ProjData/TrainHist'
preprocess_data(train_directory, train_save_directory, train_sift_directory, train_hist_directory)

# Pre-process Test data
test_directory = 'ProjData/Test'
test_save_directory = 'ProjData/TestResized'
test_sift_directory = 'ProjData/TestSift'
test_hist_directory = 'ProjData/TestHist'
preprocess_data(test_directory, test_save_directory, test_sift_directory, test_hist_directory)

# 4C. KNN Classifier
def load_hist_data(hist_data_directory):
    data = []
    labels = []
    for subdir in os.listdir(hist_data_directory):
        subdir_path = os.path.join(hist_data_directory, subdir)
        if os.path.isdir(subdir_path):
            for hist_filename in os.listdir(subdir_path):
                hist_path = os.path.join(subdir_path, hist_filename)
                hist_data = np.load(hist_path).flatten()
                data.append(hist_data)
                labels.append(subdir)
    return data, labels

train_hist_data, train_labels = load_hist_data(test_hist_directory)
test_hist_data, test_labels = load_hist_data('ProjData/TrainHist')

train_hist_data = normalize(train_hist_data, norm='l1')
test_hist_data = normalize(test_hist_data, norm='l1')


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_hist_data, train_labels)


predicted_labels = knn.predict(test_hist_data)
print(classification_report(test_labels, predicted_labels))

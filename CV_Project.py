import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import collections
from sklearn.svm import LinearSVC
import shutil


min_keypoints = 10
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
                    if descriptor is not None and descriptor.shape[0] >= min_keypoints:  # Check if the descriptor array is not empty and has enough keypoints
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



# 4A 50*50 KNN Classifier
train_features = [] 
train_labels = [] 
 
# Retrieving all the 50x50 train images from the resized directory
for subdir in os.listdir(train_save_directory): 
    subdir_path = os.path.join(train_save_directory, subdir) 
    if os.path.isdir(subdir_path): 
        for img_name in os.listdir(subdir_path): 
            img_path = os.path.join(subdir_path, img_name) 
            if "50x50" in img_name: 
                img = cv2.imread(img_path)
                feature_vector = img.flatten()
                train_features.append(feature_vector)
                train_labels.append(subdir)

test_features = []
test_labels = []

# Retrieving all the 50x50 test images from the resized directory
for subdir in os.listdir(test_save_directory):
    subdir_path = os.path.join(test_save_directory, subdir)
    if os.path.isdir(subdir_path):
        for img_name in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_name)
            if "50x50" in img_name:  
                img = cv2.imread(img_path)
                feature_vector = img.flatten()
                test_features.append(feature_vector)
                test_labels.append(subdir)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features, train_labels)

test_predictions = knn.predict(test_features) 
print("Classification report for 4A:\n")
print(classification_report(test_labels, test_predictions)) 


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
print("Classification report for 4C:\n")
print(classification_report(test_labels, predicted_labels))


# 4D. SVM Classifier
def load_sift_data(sift_directory):
    X = []
    y = []
    for subdir in os.listdir(sift_directory):
        subdir_path = os.path.join(sift_directory, subdir)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                descriptor = np.load(file_path, allow_pickle=True) #added allow pickle
                X.append(descriptor)
                y.append(subdir)
    return X, y

# Stack descriptors and create a label array
def stack_descriptors_and_labels(X, y):
    stacked_descriptors = np.vstack(X)
    labels = np.hstack([[label] * len(descriptors) for label, descriptors in zip(y, X)])
    return stacked_descriptors, labels

# Load SIFT features and labels for training and testing
X_train, y_train = load_sift_data(train_sift_directory)
X_test, y_test = load_sift_data(test_sift_directory)

# Stack descriptors and create label arrays
X_train_stacked, y_train_stacked = stack_descriptors_and_labels(X_train, y_train)
X_test_stacked, y_test_stacked = stack_descriptors_and_labels(X_test, y_test)

# Train a linear SVM classifier
clf = LinearSVC()
clf.fit(X_train_stacked, y_train_stacked)

# Predict on test data
y_pred = clf.predict(X_test_stacked)
print("Classification report for 4D:\n")
# Evaluate the performance of the classifier
print(classification_report(y_test_stacked, y_pred))


#4B SIFT features KNN
A_train, B_train = load_sift_data(train_sift_directory)
A_test, B_test = load_sift_data(test_sift_directory)

A_train_stacked, B_train_stacked = stack_descriptors_and_labels(X_train, y_train)
A_test_stacked, B_test_stacked = stack_descriptors_and_labels(X_test, y_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(A_train_stacked, B_train_stacked)

y_pred = knn.predict(A_test_stacked)
report = classification_report(B_test_stacked, y_pred)
print("Classification report for 4B:\n")
print(report)

# Remode the directories
shutil.rmtree('ProjData/TrainResized')
shutil.rmtree('ProjData/TrainSift')
shutil.rmtree('ProjData/TrainHist')
shutil.rmtree('ProjData/TestResized')
shutil.rmtree('ProjData/TestSift')
shutil.rmtree('ProjData/TestHist')

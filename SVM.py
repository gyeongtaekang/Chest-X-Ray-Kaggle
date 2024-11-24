import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

dataset_path = r"C:\Users\AERO\Downloads\archive (1)\chest_xray"
train_direct = os.path.join(dataset_path, "train")
labels = ['NORMAL', 'PNEUMONIA']

# 각 클래스의 이미지 수 확인
train_normal = os.path.join(train_direct, "NORMAL")
train_pneumonia = os.path.join(train_direct, "PNEUMONIA")
print("no.PNEUMONIA:", len(os.listdir(train_pneumonia)))
print("no.NORMAL:", len(os.listdir(train_normal)))

# 샘플 이미지 확인
sample_normal = os.listdir(train_normal)[0]
sample_pneumonia = os.listdir(train_pneumonia)[0]
normal_image = os.path.join(train_normal, sample_normal)
pneumonia_image = os.path.join(train_pneumonia, sample_pneumonia)

# 이미지 시각화
img_normal = cv2.imread(normal_image)
img_pneumonia = cv2.imread(pneumonia_image)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB))
plt.title("Normal Image")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_pneumonia, cv2.COLOR_BGR2RGB))
plt.title("Pneumonia Image")
plt.show()

# 데이터 준비
data = []
target = []
for label in labels:
    path = os.path.join(train_direct, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            resized_img = cv2.resize(img, (128, 128))
            data.append(resized_img)
            target.append(labels.index(label))

# NumPy 배열로 변환
data = np.array(data)
target = np.array(target)

# 데이터 평탄화 (SVM용)
data_flat = data.reshape(data.shape[0], -1)

# SVM 모델 학습
x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(data_flat, target, test_size=0.2, random_state=42)
model_svm = SVC(max_iter=10000)
model_svm.fit(x_train_svm, y_train_svm)
y_pred_svm = model_svm.predict(x_test_svm)
accuracy_svm = accuracy_score(y_test_svm, y_pred_svm)
print("SVM Accuracy:", accuracy_svm * 100, "%")

# PCA 적용 후 SVM 학습
pca = PCA(0.95)
data_pca = pca.fit_transform(data_flat)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(data_pca, target, test_size=0.2, random_state=42)
model_pca_svm = SVC(max_iter=10000)
model_pca_svm.fit(x_train_pca, y_train_pca)
y_pred_pca_svm = model_pca_svm.predict(x_test_pca)
accuracy_pca_svm = accuracy_score(y_test_pca, y_pred_pca_svm)
print("PCA + SVM Accuracy:", accuracy_pca_svm * 100, "%")

# 신경망 모델 생성
model_nn = Sequential([
    Conv2D(64, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(255, activation="relu"),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 신경망 모델 학습
x_train_nn, x_test_nn, y_train_nn, y_test_nn = train_test_split(data, target, test_size=0.3, random_state=42)
history = model_nn.fit(x_train_nn, y_train_nn, epochs=10, batch_size=8)

# 신경망 모델 평가
nn_loss, nn_accuracy = model_nn.evaluate(x_test_nn, y_test_nn)
print("Neural Network Accuracy:", nn_accuracy * 100, "%")

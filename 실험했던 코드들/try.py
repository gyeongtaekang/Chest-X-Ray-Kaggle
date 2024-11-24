# data processing, CSV & image file I/O
import os
import re
import requests
from PIL import Image
import pandas as pd
import numpy as np

#libraries for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Deep learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import warnings
warnings.filterwarnings('ignore')

# Function to generate data paths with labels
def generate_dataframe(data_dir):
    filepaths = []
    labels = []
    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

# Generate train, validation, and test dataframes
train_data_dir = r'C:\Users\AERO\Downloads\archive (1)\chest_xray/train'
val_data_dir = r'C:\Users\AERO\Downloads\archive (1)\chest_xray/val'
test_data_dir = r'C:\Users\AERO\Downloads\archive (1)\chest_xray/test'

train_df = generate_dataframe(train_data_dir)
valid_df = generate_dataframe(val_data_dir)
test_df = generate_dataframe(test_data_dir)

# ImageDataGenerator for train, validation, and test datasets
batch_size = 16
img_size = (224, 224)

tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()
val_gen = ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)

valid_gen = val_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                        color_mode='rgb', shuffle=True, batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                       color_mode='rgb', shuffle=False, batch_size=batch_size)

# Plot a batch of training images
g_dict = train_gen.class_indices
classes = list(g_dict.keys())
images, labels = next(train_gen)

plt.figure(figsize=(20, 20))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    image = images[i] / 255
    plt.imshow(image)
    index = np.argmax(labels[i])
    class_name = classes[index]
    plt.title(class_name, color='blue', fontsize=12)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Define the model architecture
img_shape = (img_size[0], img_size[1], 3)
class_count = len(classes)

model = Sequential([
    Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", input_shape=img_shape),
    Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPool2D((2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPool2D((2, 2)),
    Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(64, activation="relu"),
    Dense(class_count, activation="softmax")
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
epochs = 10
history = model.fit(x=train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=False)

# Plot training history
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

Epochs = [i + 1 for i in range(len(tr_acc))]
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]

plt.figure(figsize=(20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=f'best epoch= {index_loss + 1}')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=f'best epoch= {index_acc + 1}')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
train_score = model.evaluate(train_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
test_score = model.evaluate(test_gen, verbose=1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

# Generate classification report and confusion matrix
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
print(classification_report(test_gen.classes, y_pred, target_names=classes))

cm = confusion_matrix(test_gen.classes, y_pred)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Save the model
model.save('Pneumonia.h5')
loaded_model = tf.keras.models.load_model('Pneumonia.h5', compile=False)
loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Predict on a single image
image_path = 'your_image_path_here.jpeg'
image = Image.open(image_path)
img = image.resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = loaded_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(f"{classes[tf.argmax(score)]}")


# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 224, 224, 64)        │           1,792 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 224, 224, 64)        │          36,928 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 112, 112, 64)        │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_2 (Conv2D)                    │ (None, 112, 112, 128)       │          73,856 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_3 (Conv2D)                    │ (None, 112, 112, 128)       │         147,584 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 56, 56, 128)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_4 (Conv2D)                    │ (None, 56, 56, 256)         │         295,168 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_5 (Conv2D)                    │ (None, 56, 56, 256)         │         590,080 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_6 (Conv2D)                    │ (None, 56, 56, 256)         │         590,080 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_2 (MaxPooling2D)       │ (None, 28, 28, 256)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 200704)              │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 256)                 │      51,380,480 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 64)                  │          16,448 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 2)                   │             130 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 53,132,546 (202.68 MB)
#  Trainable params: 53,132,546 (202.68 MB)
#  Non-trainable params: 0 (0.00 B)
# Epoch 1/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 603s 2s/step - accuracy: 0.8593 - loss: 15.2764 - val_accuracy: 0.8750 - val_loss: 0.2830
# Epoch 2/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 587s 2s/step - accuracy: 0.9603 - loss: 0.1112 - val_accuracy: 0.8750 - val_loss: 0.2088
# Epoch 3/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 570s 2s/step - accuracy: 0.9744 - loss: 0.0655 - val_accuracy: 0.6250 - val_loss: 1.0888
# Epoch 4/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 580s 2s/step - accuracy: 0.9746 - loss: 0.0647 - val_accuracy: 0.8750 - val_loss: 0.1678
# Epoch 5/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 555s 2s/step - accuracy: 0.9850 - loss: 0.0419 - val_accuracy: 0.8750 - val_loss: 0.1978
# Epoch 6/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 569s 2s/step - accuracy: 0.9899 - loss: 0.0327 - val_accuracy: 1.0000 - val_loss: 0.0098
# Epoch 7/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 571s 2s/step - accuracy: 0.9907 - loss: 0.0262 - val_accuracy: 1.0000 - val_loss: 9.5406e-04
# Epoch 8/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 581s 2s/step - accuracy: 0.9937 - loss: 0.0166 - val_accuracy: 0.8750 - val_loss: 0.2392
# Epoch 9/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 575s 2s/step - accuracy: 0.9960 - loss: 0.0088 - val_accuracy: 0.8125 - val_loss: 0.7549
# Epoch 10/10
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 595s 2s/step - accuracy: 0.9931 - loss: 0.0155 - val_accuracy: 0.8750 - val_loss: 0.3079
# 326/326 ━━━━━━━━━━━━━━━━━━━━ 165s 507ms/step - accuracy: 0.9980 - loss: 0.0067
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 577ms/step - accuracy: 0.8750 - loss: 0.3079
# 39/39 ━━━━━━━━━━━━━━━━━━━━ 20s 499ms/step - accuracy: 0.4906 - loss: 5.2650
# Train Loss:  0.008485114201903343
# Train Accuracy:  0.9973159432411194
# --------------------
# Validation Loss:  0.3078680634498596
# Validation Accuracy:  0.875
# --------------------
# Test Loss:  2.923175096511841
# Test Accuracy:  0.7259615659713745
# 39/39 ━━━━━━━━━━━━━━━━━━━━ 20s 503ms/step
#               precision    recall  f1-score   support

#       NORMAL       0.94      0.29      0.44       234
#    PNEUMONIA       0.70      0.99      0.82       390

#     accuracy                           0.73       624
#    macro avg       0.82      0.64      0.63       624
# weighted avg       0.79      0.73      0.68       624

# Traceback (most recent call last):
#   File "c:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main\try.py", line 183, in <module>
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#                 ^^^^^^^^^
# NameError: name 'itertools' is not defined. Did you forget to import 'itertools'?
# PS C:\Users\AERO\Downloads\Chest-X-Ray-Kaggle-main\Chest-X-Ray-Kaggle-main> 
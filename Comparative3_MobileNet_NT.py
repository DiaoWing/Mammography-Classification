import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers
from keras.applications import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.regularizers import l2  
import random

# 设置所有随机种子以实现可复现性
SEED = 42  # 你可以选择任何整数作为种子值

# 1. 设置Python随机种子
random.seed(SEED)

# 2. 设置NumPy随机种子
np.random.seed(SEED)

# 3. 设置TensorFlow随机种子
tf.random.set_seed(SEED)

# 定义一个变量来记录最高验证准确率
best_val_acc = 0.0

######################### Step1: Initialize parameters and file paths #########################
# Configuration
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 40  #25 ，40 ，50
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 120
CHANNELS = 3

# File paths
MODEL_FOLDER = 'MobileNet_NT_Model'
BASE_DIR = 'E:/StudyInformation/Graduation/DegreeThesis/FormalPaper/codes/MIAS-master/split_MIAS_ROIs_data'

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

# Create output directories
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(os.path.join(MODEL_FOLDER, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(MODEL_FOLDER, 'plots'), exist_ok=True)
os.makedirs(os.path.join(MODEL_FOLDER, 'history'), exist_ok=True)

######################### Step2: Data preparation #########################
# Modern data augmentation with preprocessing
train_datagen = ImageDataGenerator(
    rotation_range=180,  #旋转0-180
    zoom_range=0.2,      #缩放0.2
    width_shift_range = 0.1,  #水平移动过范围
    height_shift_range = 0.1, #垂直移动范围
    shear_range = 0.2,        #裁剪
    horizontal_flip=True,     #是否翻转
    fill_mode='nearest',      #填充方式
    samplewise_center=True,
)

val_datagen = ImageDataGenerator(samplewise_center=True)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

######################### Step3: Model construction #########################
# Load pre-trained MobileNet with modern practices
base_model = MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
    alpha=1.0  # Width multiplier (1.0 for full MobileNet)
)

# Freeze base model layers
base_model.trainable = False

# Add custom head with modern practices
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.6)(x)  # Added dropout for regularization
x = layers.Dense(256, activation='relu',kernel_regularizer=l2(0.001))(x)     #
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Modern optimizer configuration
optimizer = optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

######################### Step4: Training #########################
# Modern callbacks
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_FOLDER, 'checkpoints', 'weights-improvement-best.hdf5'),
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
# 创建打印最高验证准确率的回调函数
print_best_val_acc = LambdaCallback(
    on_epoch_end=lambda epoch, logs: globals().update({
        'best_val_acc': max(logs.get('val_accuracy', 0), best_val_acc)
    }) or print(f'\nCurrent best validation accuracy: {best_val_acc:.4f}')
)


# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print('Training starts!\n')

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, print_best_val_acc],# early_stopping
    verbose=2
)

######################### Step5: Save results and visualization #########################
# Save history in numpy format
history_path = os.path.join(MODEL_FOLDER, 'history', 'history_Baseline.npy')
np.save(history_path, history.history)

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'bo-', label='Training Loss')
plt.plot(history.history['val_loss'], 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_FOLDER, 'plots', 'training_history.jpg'))
plt.show()
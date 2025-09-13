import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2  


# 定义一个变量来记录最高验证准确率
best_val_acc = 0.0

######################### Step1: Initialize parameters and file paths #########################
# Configuration
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 40  #25 ，40 ，50
IMAGE_HEIGHT = 70
IMAGE_WIDTH = 70
CHANNELS = 3

# File paths
MODEL_FOLDER = 'DenseNet_NT_Model'
BASE_DIR = 'E:/StudyInformation/Graduation/DegreeThesis/FormalPaper/codes/MIAS-master/split_MIAS_ROIs_data'

TRAIN_DIR = os.path.join(BASE_DIR, 'train_aug_slip')
VAL_DIR = os.path.join(BASE_DIR, 'val_aug_slip')

# Create output directories
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(os.path.join(MODEL_FOLDER, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(MODEL_FOLDER, 'plots'), exist_ok=True)
os.makedirs(os.path.join(MODEL_FOLDER, 'history'), exist_ok=True)

######################### Step2: Data preparation #########################
# Using samplewise_center=True instead of rescale=1./255 to match other implementations
train_datagen = ImageDataGenerator(
    # rotation_range=180,  #旋转0-180
    # zoom_range=0.2,      #缩放0.2
    # width_shift_range = 0.1,  #水平移动过范围
    # height_shift_range = 0.1, #垂直移动范围
    # shear_range = 0.2,        #裁剪
    # horizontal_flip=True,     #是否翻转
    # fill_mode='nearest',      #填充方式
    samplewise_center=True,   #能够进行归一话处理
)

val_datagen = ImageDataGenerator(
    samplewise_center=True
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',   #将分类转化为ONE-HOT编码格式
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
# Load pre-trained DenseNet121   #在imagenet上训练好的大模型，mobilenet，
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,     #意味着分类头我们可以对其进行微调
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)   #3 彩色 1 灰色
)

# Freeze base model layers
base_model.trainable = False        #前面的层不在训练，不更新参数    构建模型使用的语法有三种

# Add custom head (consistent with other implementations)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.BatchNormalization()(x)
x = layers.Dense(1024, activation='relu')(x)       #1024。fc为口语化的说法
x = layers.Dropout(0.5)(x)       
# x = layers.Dense(1024, activation='relu')(x)       #1024
x = layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)  # L2正则化，正则化是防止过拟合
# x = layers.Dense(512,activation = 'relu')(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)            #对其进行分类，分类数量，2，3，0-1之间的数  

model = Model(inputs=base_model.input, outputs=outputs)
# 加载DenseNet121，输入形状改为单通道

# # 定义ReduceLROnPlateau回调
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.2,     # 学习率乘以0.2
#     patience=3,     # 等待3个epoch无改善
#     min_lr=1e-5,    # 最小学习率
#     verbose=1       # 打印日志
# )


optimizer =optimizers.Adam(learning_rate=1e-4) #（接收预测值和标签）      # optimizers.SGD(learning_rate=0.0001, momentum=0.9)#optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',      #计算损失的函数，面对不同的问题会使用不同的函数
    metrics=['accuracy']       #保存验证集准确率较高的模型
)

model.summary()

######################### Step4: Training #########################
# Callbacks
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
    callbacks=[checkpoint, print_best_val_acc],#early_stopping，reduce_lr, 
    verbose=2
)

######################### Step5: Save results #########################
# Save history
history_path = os.path.join(MODEL_FOLDER, 'history', 'history_Baseline.npy')  #.csv则则保存为其他格式
np.save(history_path, history.history)

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'bo-', label='Training Accuracy')   # 训练集的损失
plt.plot(history.history['val_accuracy'], 'ro-', label='Validation Accuracy')  #验证集的损失
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
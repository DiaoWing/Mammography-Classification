import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from itertools import cycle

#config
BASE_DIR = "E:/StudyInformation/Graduation/DegreeThesis/FormalPaper/codes/MIAS-master/split_MIAS_ROIs_data"
MODEL_FOLDER = "E:/StudyInformation/Graduation/DegreeThesis/FormalPaper/codes/对比2_aug/NO.2_D/DenseNet_NT_Model_动态2/checkpoints"
TRAIN_DIR = "E:/StudyInformation/Graduation/DegreeThesis/FormalPaper/codes/MIAS-master/split_MIAS_ROIs_data/train"
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 40   #25 ，40 ，50
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 120
CHANNELS = 3

# 配置参数
TEST_DIR = os.path.join(BASE_DIR, 'test')  # 测试集路径
BEST_MODEL_PATH = os.path.join(MODEL_FOLDER,  'weights-improvement-best.hdf5')


train_datagen = ImageDataGenerator(
    rotation_range=180,  #旋转0-180
    zoom_range=0.2,      #缩放0.2
    width_shift_range = 0.1,  #水平移动过范围
    height_shift_range = 0.1, #垂直移动范围
    shear_range = 0.2,        #裁剪
    horizontal_flip=True,     #是否翻转
    fill_mode='nearest',      #填充方式
    samplewise_center=True,   #能够进行归一话处理
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
CLASS_NAMES = list(train_generator.class_indices.keys())  # 获取类别名称



# 加载最佳模型
model = load_model(BEST_MODEL_PATH)

# 准备测试数据生成器
test_datagen = ImageDataGenerator(samplewise_center=True)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # 测试集不需要打乱
)

# 获取真实标签和预测结果
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

# 1. 计算并打印分类报告
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# 2. 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)

# 3. 绘制ROC曲线
def plot_roc_curve(y_true, y_pred_prob, classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    
    # 计算每个类别的ROC曲线和AUC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制所有类别的ROC曲线
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(y_true, y_pred_prob, CLASS_NAMES)

# 4. 计算并显示主要指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\nKey Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 5. 分析预测错误的样本（可选）
def analyze_errors(y_true, y_pred, test_generator, num_samples=5):
    errors = np.where(y_pred != y_true)[0]
    print(f"\nTotal misclassified samples: {len(errors)}/{len(y_true)}")
    
    # 显示部分错误分类样本
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(errors[:num_samples]):
        img_path = test_generator.filepaths[idx]
        img = plt.imread(img_path)
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(f"True: {CLASS_NAMES[y_true[idx]]}\nPred: {CLASS_NAMES[y_pred[idx]]}")
        plt.axis('off')
    plt.show()

analyze_errors(y_true, y_pred, test_generator)
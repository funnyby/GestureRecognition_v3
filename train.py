import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# 设置GPU内存增长
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

def load_data(csv_path):
    # 读取CSV文件
    print(f"正在从 {csv_path} 加载数据...")
    df = pd.read_csv(csv_path)
    
    # 分离标签和像素数据
    labels = df['label'].values
    images = df.drop('label', axis=1).values
    
    # 重塑图像数据为28x28的格式
    images = images.reshape(-1, 28, 28, 1)
    
    # 归一化像素值
    images = images.astype('float32') / 255.0
    
    print(f"已加载 {len(images)} 个样本")
    return images, labels

def getCNNModel(shape):
    model = Sequential([
        Conv2D(64, (3, 3), strides=1, padding="same", input_shape=shape),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),
        
        Conv2D(128, (3, 3), strides=1, padding='same'),
        LeakyReLU(alpha=0.1),
        Dropout(0.25),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding='same'),
        
        Conv2D(64, (3, 3), strides=1, padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding='same'),
        
        Conv2D(32, (3, 3), strides=1, padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        
        Flatten(),
        Dense(512),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(units=25, activation='softmax')
    ])
    return model

def train_model(model_dir='./models', verbose=1):
    # 使用绝对路径
    train_data_dir = 'E:/xianyudaizuo/bishe/bishe3/GestureRecognition/sign_mnist_train.csv'
    test_data_dir = 'E:/xianyudaizuo/bishe/bishe3/GestureRecognition/sign_mnist_test.csv'
    
    # 创建模型保存目录
    os.makedirs(model_dir, exist_ok=True)

    # 加载数据
    X_train, y_train = load_data(train_data_dir)
    X_test, y_test = load_data(test_data_dir)

    # 检查数据是否正确加载
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("无法在指定目录中找到图像。")

    # 重塑数据
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # 输出形状信息
    print(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")
    print(f"训练标签形状: {y_train.shape}, 测试标签形状: {y_test.shape}")

    # 创建并编译模型
    model = getCNNModel(X_train[0].shape)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    if verbose > 0:
        model.summary()

    # 定义回调函数
    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=verbose, factor=0.5, min_lr=0.00001),
        EarlyStopping(monitor='val_loss', patience=10, verbose=verbose, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            verbose=verbose,
            save_best_only=True,
            mode='max'
        )
    ]

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 数据增强
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        horizontal_flip=False,
        vertical_flip=False
    )
    datagen.fit(X_train)

    # 训练模型
    batch_size = 64
    epochs = 30
    
    print("开始训练模型...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose
    )

    # 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=verbose)
    print(f"测试准确率: {test_acc*100:.2f}%")

    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"模型已保存到 {final_model_path}")
    
    # 保存一个特殊路径的模型用于UI集成
    ui_model_path = 'E:/xianyudaizuo/bishe/bishe8-改进手势识别界面/GestureRecognition/UI_rec/models/cnn_model.h5'
    os.makedirs(os.path.dirname(ui_model_path), exist_ok=True)
    model.save(ui_model_path)
    
    return model, history

# 仅当直接运行此脚本时训练模型
if __name__ == "__main__":
    model, history = train_model(verbose=1)

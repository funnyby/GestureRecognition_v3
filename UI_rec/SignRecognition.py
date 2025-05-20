# -*- coding: utf-8 -*-

import time
import numpy as np
import cv2
import mediapipe as mp
import os
import random
import tensorflow as tf
from numberr import get_str_guester

class Sign_Recognition:
    def __init__(self):
        self.CAM_NUM = 1  # 摄像头标号
        self.cap = None  # 摄像头对象
        self.current_image = None
        self.detInfo = []
        
        # 初始化摄像头
        try:
            self.cap = cv2.VideoCapture(self.CAM_NUM)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
        except Exception as e:
            print("摄像头初始化失败:", e)
            return
            
        # 初始化手势识别模型
        try:
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mpDraw = mp.solutions.drawing_utils
        except Exception as e:
            print("手势模型初始化失败:", e)
            return
            
        # 加载CNN模型
        self._load_cnn_model()

    def _load_cnn_model(self):
        try:
            model_path = os.path.join('models', 'cnn_model.h5')
            # 这里可以添加模型加载代码
            pass
        except Exception as e:
            print("模型加载失败:", e)

    def start_camera(self):
        """启动摄像头识别"""
        if not self.cap or not self.cap.isOpened():
            print("摄像头未打开")
            return
            
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                    
                # 处理图像
                self.process_frame(frame)
                
                # 显示图像
                cv2.imshow('手势识别', frame)
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print("处理过程中出错:", e)
                break
                
        self.cleanup()

    def process_frame(self, frame):
        """处理单帧图像"""
        try:
            # 转换为RGB
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 进行手势识别
            results = self.hands.process(imgRGB)
            
            if results.multi_hand_landmarks:
                self.detInfo = []
                image_height, image_width, _ = frame.shape
                
                for hand in results.multi_hand_landmarks:
                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        list_lms.append([int(pos_x), int(pos_y)])

                    # 构造凸包点
                    list_lms = np.array(list_lms, dtype=np.int32)
                    
                    # 区域位置
                    xmin = list_lms[:, 0].min() - 20
                    ymin = list_lms[:, 1].min() - 20
                    xmax = list_lms[:, 0].max() + 20
                    ymax = list_lms[:, 1].max() + 20
                    bbox = [xmin, ymin, xmax, ymax]

                    # 计算手势
                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms[hull_index, :])

                    # 查找外部的点数
                    ll = [4, 8, 12, 16, 20]
                    up_fingers = []

                    for i in ll:
                        pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                        dist = cv2.pointPolygonTest(hull, pt, True)
                        if dist < 0:
                            up_fingers.append(i)

                    str_guester = get_str_guester(up_fingers, list_lms)
                    self.detInfo.append([str_guester, bbox])

                    # 绘制关键点和连接线
                    for i in ll:
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        cv2.circle(frame, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)

                    cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
                    self.mpDraw.draw_landmarks(frame, hand, self.mpHands.HAND_CONNECTIONS)

                    # 绘制边界框和标签
                    self.draw_rect_box(frame, bbox, str_guester)
                    
                    # 打印识别结果
                    print(f"识别到手势: {str_guester}")
                    
        except Exception as e:
            print("处理帧时出错:", e)

    def draw_rect_box(self, image, rect, text):
        """绘制边界框和标签"""
        try:
            cv2.rectangle(image, 
                         (int(round(rect[0])), int(round(rect[1]))),
                         (int(round(rect[2])), int(round(rect[3]))),
                         (0, 0, 255), 2)
            cv2.rectangle(image, 
                         (int(rect[0] - 1), int(rect[1]) - 20), 
                         (int(rect[0] + 120), int(rect[1])), 
                         (0, 0, 255), -1)
            cv2.putText(image, text, 
                       (int(rect[0] + 1), int(rect[1] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print("绘制边界框时出错:", e)

    def cleanup(self):
        """清理资源"""
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # 释放模型资源
            if hasattr(self, 'cnn_model') and self.cnn_model is not None:
                self.cnn_model = None
                try:
                    tf.keras.backend.clear_session()
                except Exception:
                    pass
        except Exception as e:
            print("清理资源时出错:", e)
# -*- coding: utf-8 -*-

import os
import warnings
import cv2
import mediapipe as mp
import numpy as np
from numberr import get_str_guester

class GestureRecognizer:
    def __init__(self, camera_id=0):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")

        # 初始化手势识别模型
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils

    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            return None, None

        image_height, image_width, _ = np.shape(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        gestures = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 收集所有关键点的坐标
                list_lms = []
                for i in range(21):
                    pos_x = hand_landmarks.landmark[i].x * image_width
                    pos_y = hand_landmarks.landmark[i].y * image_height
                    list_lms.append([int(pos_x), int(pos_y)])

                # 构建凸包点
                list_lms = np.array(list_lms, dtype=np.int32)
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

                # 获取手势
                gesture = get_str_guester(up_fingers, list_lms)
                gestures.append(gesture)

        return img, gestures

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.hands is not None:
            self.hands.close()

def main():
    # 忽略警告
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings(action='ignore')

    try:
        # 创建手势识别器实例
        recognizer = GestureRecognizer(camera_id=0)
        
        print("手势识别已启动，按'q'退出...")
        
        # 创建窗口并设置关闭回调
        cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gesture Recognition", cv2.WND_PROP_TOPMOST, 1)
        
        while True:
            # 处理帧
            frame, gestures = recognizer.process_frame()
            
            if frame is not None:
                # 显示结果
                if gestures:
                    print(f"检测到手势: {gestures}")
                
                # 显示图像
                cv2.imshow("Gesture Recognition", frame)
                
                # 检查窗口是否被关闭
                if cv2.getWindowProperty("Gesture Recognition", cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 释放资源
        if 'recognizer' in locals():
            recognizer.release()
        cv2.destroyAllWindows()
        # 确保所有窗口都被关闭
        for i in range(5):
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
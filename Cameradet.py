import mediapipe as mp
import cv2
import time
import math

class GestureDetector:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.gesture = [0, 1, 2, 3, 4, 5]

    def _calculate_gesture(self, hand_landmarks):
        flag = 0
        p0_x = hand_landmarks.landmark[0].x
        p0_y = hand_landmarks.landmark[0].y
        p5_x = hand_landmarks.landmark[5].x
        p5_y = hand_landmarks.landmark[5].y
        distance_0_5 = math.sqrt(pow(p0_x - p5_x, 2) + pow(p0_y - p5_y, 2))
        base = distance_0_5 / 0.6

        p4_x = hand_landmarks.landmark[4].x
        p4_y = hand_landmarks.landmark[4].y
        distance_5_4 = math.sqrt(pow(p5_x - p4_x, 2) + pow(p5_y - p4_y, 2))

        p8_x = hand_landmarks.landmark[8].x
        p8_y = hand_landmarks.landmark[8].y
        distance_0_8 = math.sqrt(pow(p0_x - p8_x, 2) + pow(p0_y - p8_y, 2))

        p12_x = hand_landmarks.landmark[12].x
        p12_y = hand_landmarks.landmark[12].y
        distance_0_12 = math.sqrt(pow(p0_x - p12_x, 2) + pow(p0_y - p12_y, 2))

        p16_x = hand_landmarks.landmark[16].x
        p16_y = hand_landmarks.landmark[16].y
        distance_0_16 = math.sqrt(pow(p0_x - p16_x, 2) + pow(p0_y - p16_y, 2))

        p20_x = hand_landmarks.landmark[20].x
        p20_y = hand_landmarks.landmark[20].y
        distance_0_20 = math.sqrt(pow(p0_x - p20_x, 2) + pow(p0_y - p20_y, 2))

        if distance_0_8 > base:
            flag += 1
        if distance_0_12 > base:
            flag += 1
        if distance_0_16 > base:
            flag += 1
        if distance_0_20 > base:
            flag += 1
        if distance_5_4 > base * 0.2:
            flag += 1

        return self.gesture[flag]

    def detect_gesture(self, frame=None):
        """
        �������
        :param frame: ����֡�����ΪNone�������ͷ��ȡ
        :return: �ֵ䣬���������ֵ�ʶ����
        """
        if frame is None:
            success, frame = self.cap.read()
            if not success:
                return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        dict_handnumber = {}
        if results.multi_handedness:
            if len(results.multi_handedness) == 2:
                for i in range(len(results.multi_handedness)):
                    label = results.multi_handedness[i].classification[0].label
                    index = results.multi_handedness[i].classification[0].index
                    hand_landmarks = results.multi_hand_landmarks[index]
                    gesresult = self._calculate_gesture(hand_landmarks)
                    dict_handnumber[label] = gesresult
            else:
                label = results.multi_handedness[0].classification[0].label
                hand_landmarks = results.multi_hand_landmarks[0]
                gesresult = self._calculate_gesture(hand_landmarks)
                dict_handnumber[label] = gesresult

        return dict_handnumber

    def release(self):
        """
        �ͷ���Դ
        """
        self.cap.release()
        self.hands.close()

    def __del__(self):
        self.release()

# ʹ��ʾ��
if __name__ == "__main__":
    detector = GestureDetector()
    try:
        while True:
            result = detector.detect_gesture()
            if result:
                print("ʶ识别结果:", result)
            time.sleep(0.1)  # ����ʶ��Ƶ��
    except KeyboardInterrupt:
        print("键盘终止")
    finally:
        detector.release()

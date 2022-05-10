import cv2
import time
import os

import mediapipe as mp

import settings_vector

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
folderPath = "number_finger"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_NEAREST)
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
pTime = 0
hands = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.9)
def_hand_length = 0.17
tipIds = [4, 8, 12, 16, 20]
fin_up = [0, 0, 0, 0, 0]
while True:
    success, img = cap.read()
    result = hands.process(img)
    fin_up = [0, 0, 0, 0, 0]
    if result.multi_hand_landmarks is not None:
        np_result = settings_vector.get_numpy_points(result.multi_hand_landmarks[0])
        rotate_result = settings_vector.set_standard_position(np_result)
        size_result = settings_vector.set_standard_size(rotate_result)
        #
        # z_o_h_l = def_hand_length / (settings_vector.get_length(rotate_result[0], rotate_result[5]))
        # rotate_result[:, :2] *= z_o_h_l
        # rotate_result[0][2] = z_o_h_l
        # rotate_result[:, 2] += z_o_h_l

        if rotate_result[tipIds[0]][0] < rotate_result[tipIds[0] - 1][0]:
            fin_up[0] = 1

        for i in range(1, len(fin_up)):
            if rotate_result[tipIds[i]][1] > rotate_result[tipIds[i] - 1][1]:
                fin_up[i] = 1

    # img = detector.findHands(img)
    # lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    # if len(lmList) != 0:
    #     fingers = []
    #     # Thumb
    #     if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
    #         fingers.append(1)
    #     else:
    #         fingers.append(0)
    #     # 4 Fingers
    #     for id in range(1, 5):
    #         if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)
        # print(fingers)
        # totalFingers = fingers.count(1)
        # print(totalFingers)
        # h, w, c = overlayList[totalFingers - 1].shape
        # img[0:h, 0:w] = overlayList[totalFingers - 1]
        # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        # cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
        #             10, (255, 0, 0), 25)
    print(fin_up.count(1))
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.putText(img, " ".join(map(str, fin_up)), (100, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

import re
import sys
import time
import json
import queue
import pickle
from threading import Thread

import cv2
import torch
import pyaudio
import mediapipe as mp
import vosk
import sounddevice as sd

from itention_pred import PredItent
# from ai_virtual_painter import virtual_painter

import cv2
import time
import hand_track_pointer as htm
import numpy as np
import os

import cv2
import time
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import settings_vector


class ClassThreading:
    """

    """

    def __init__(self, speaker_name, id_camera=0):
        # camera
        self.cap = cv2.VideoCapture(id_camera)

        # mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.model_hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.number_finger = 8
        self.count_frame_btn = 6

        # vosk
        self.language = 'ru'
        self.model_id = 'ru_v3'
        self.sample_rate = 48000  # 48000
        self.speaker = speaker_name  # aidar, baya, kseniya, xenia, random
        self.put_accent = True
        self.put_yo = True
        self.device = torch.device('cpu')  # cpu или gpu
        self.model_vosk, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                            model='silero_tts',
                                            language=self.language,
                                            speaker=self.model_id)
        self.path_to_hello = r'C:\Users\79614\PycharmProjects\diplom_\speach_comand\down_button.pkl'
        self.hello_object = self.load_object(self.path_to_hello)
        # self.what = self.hello_object

        # kaldi
        self.stt_samplerate = 16000
        self.stt_device = 1
        self.stt_model = vosk.Model(r"C:\Users\79614\PycharmProjects\diplom_\speech_asis\model")
        self.q = queue.Queue()

        # threads
        self.count_vosk = 0
        self.end_request = True
        self.thread_vosk_hello = None
        self.thread_kaldi_ = None
        self.thread_camera_pipe = None
        self.threads = []

        # button
        self.padding = 50
        self.size_btn = 100, 100
        self.color = 255

        self.model_itent = PredItent(r'C:\Users\79614\PycharmProjects\diplom_\model')
        self.itention = ''
        self.comand = ''
        self.isopened = True

    def start_vosk(self, what):
        self.thread_vosk_hello = Thread(target=self.vosk_, name=f'thread vosk {self.count_vosk}', args=(what,))
        self.thread_vosk_hello.start()
        self.threads.append(self.thread_vosk_hello)

    def start_kaldi(self, ):
        self.thread_kaldi_ = Thread(target=self.kaldi_, name='thread_kaldi', args=())
        self.thread_kaldi_.start()
        self.thread_kaldi_.join()

    def start_camera_pipe(self):
        self.thread_camera_pipe = Thread(target=self.camera_pipe, name='thread view', args=())
        self.thread_camera_pipe.start()
        self.threads.append(self.thread_camera_pipe)
        self.thread_camera_pipe.join()

    def camera_pipe(self):
        count = 0
        while self.isopened:
            success, image = self.cap.read()
            if image is None:
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model_hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks is not None:
                x, y = results.multi_hand_landmarks[0].landmark[self.number_finger].x, \
                       results.multi_hand_landmarks[0].landmark[self.number_finger].y
                x_px, y_px = image.shape[1] * x, image.shape[0] * y
                if image.shape[1] - (self.padding + self.size_btn[1]) < x_px < image.shape[1] - self.padding and \
                        image.shape[0] - (self.padding + self.size_btn[0]) < y_px < image.shape[0] - self.padding:
                    count += 1
                    if count >= self.count_frame_btn and self.end_request:
                        count = 0
                        self.end_request = False
                        print('aer')
                        self.start_vosk(self.hello_object)

                else:
                    count = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

            image[image.shape[0] - (self.padding + self.size_btn[0]):image.shape[0] - self.padding,
            image.shape[1] - (self.padding + self.size_btn[1]):image.shape[1] - self.padding] = self.color

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.cap.release()

    def q_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def kaldi_(self):
        self.comand = self.kaldi__()
        # print(self.comand)
        self.itention = self.model_itent.predict(self.comand)
        print(self.itention)
        if self.itention in ['звук', 'рисование', 'подсчёт', 'поза руки']:
            self.isopened = False
            if self.itention == 'рисование':
                virtual_painter()
            elif self.itention == 'звук':
                virtual_sound()
            # elif self.itention == 'подсчёт':
            #     virtual_counter()
            # elif self.itention == 'поза руки':
            #     virtual_detect()
            # self.thread_camera_pipe.setDaemon(False)
        # print(self.itention)
        # self.vosk_(self.itention)

    def kaldi__(self, callback=1):
        with sd.RawInputStream(samplerate=self.stt_samplerate, blocksize=8000, device=self.stt_device, dtype='int16',
                               channels=1, callback=self.q_callback):

            rec = vosk.KaldiRecognizer(self.stt_model, self.stt_samplerate)
            break_ = False
            while True:
                data = self.q.get()
                print('говори')
                if rec.AcceptWaveform(data):
                    # callback(json.loads(rec.Result())["text"])
                    # print(rec.Result())
                    a = rec.Result()
                    f, s = [m.start() for m in re.finditer('\"', a)][-2:][0], \
                           [m.start() for m in re.finditer('\"', a)][-2:][1]
                    str_ret = a[f + 1:s]
                    break_ = True
                if break_:
                    break
        # print(str_ret)
        return str_ret

    def vosk_(self, what):
        while self.isopened:
            if isinstance(what, str):
                audio = self.model_vosk.apply_tts(text=what + "..",
                                                  speaker=self.speaker,
                                                  sample_rate=self.sample_rate,
                                                  put_accent=self.put_accent,
                                                  put_yo=self.put_yo)
            else:
                audio = what
            sd.play(audio, self.sample_rate * 1.05)
            time.sleep((len(audio) / self.sample_rate) + 0.5)
            sd.stop()
            self.end_request = True
            self.start_kaldi()
            break

    @staticmethod
    def load_object(path_to_data):
        with open(path_to_data, 'rb') as inp:
            object_pkl = pickle.load(inp)
        return object_pkl

    @staticmethod
    def save_object(path_to_save: str, obj_save: object):
        with open(path_to_save, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj_save, outp, pickle.HIGHEST_PROTOCOL)


def virtual_painter():
    overlayList = []  # list to store all the images

    brushThickness = 25
    eraserThickness = 100
    drawColor = (255, 0, 255)  # setting purple color

    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # defining canvas

    # images in header folder
    folderPath = "Header"
    myList = os.listdir(folderPath)  # getting all the images used in code
    # print(myList)
    for imPath in myList:  # reading all the images from the folder
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)  # inserting images one by one in the overlayList
    header = overlayList[0]  # storing 1st image
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # width
    cap.set(4, 720)  # height

    detector = htm.handDetector(detectionCon=0.50, maxHands=1)  # making object

    while True:

        # 1. Import image
        success, img = cap.read()
        img = cv2.flip(img, 1)  # for neglecting mirror inversion

        # 2. Find Hand Landmarks
        img = detector.findHands(img)  # using functions fo connecting landmarks
        lmList, bbox = detector.findPosition(img,
                                             draw=False)  # using function to find specific landmark position,draw false means no circles on landmarks

        if len(lmList) != 0:
            # print(lmList)
            x1, y1 = lmList[8][1], lmList[8][2]  # tip of index finger
            x2, y2 = lmList[12][1], lmList[12][2]  # tip of middle finger

            # 3. Check which fingers are up
            fingers = detector.fingersUp()
            # print(fingers)

            # 4. If Selection Mode - Two finger are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                if 100 < x1 < 200 and 200 < y1 < 300:
                    break
                # print("Selection Mode")
                # checking for click
                if y1 < 125:
                    if 250 < x1 < 450:  # if i m clicking at purple brush
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:  # if i m clicking at blue brush
                        header = overlayList[1]
                        drawColor = (255, 0, 0)
                    elif 800 < x1 < 950:  # if i m clicking at green brush
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:  # if i m clicking at eraser
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor,
                              cv2.FILLED)  # selection mode is represented as rectangle

            # 5. If Drawing Mode - Index finger is up
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)  # drawing mode is represented as circle
                # print("Drawing Mode")
                if xp == 0 and yp == 0:  # initially xp and yp will be at 0,0 so it will draw a line from 0,0 to whichever point our tip is at
                    xp, yp = x1, y1  # so to avoid that we set xp=x1 and yp=y1
                # till now we are creating our drawing but it gets removed as everytime our frames are updating so we have to define our canvas where we can draw and show also

                # eraser
                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor,
                             brushThickness)  # gonna draw lines from previous coodinates to new positions
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1  # giving values to xp,yp everytime

            # merging two windows into one imgcanvas and img

        # 1 converting img to gray
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

        # 2 converting into binary image and thn inverting
        _, imgInv = cv2.threshold(imgGray, 50, 255,
                                  cv2.THRESH_BINARY_INV)  # on canvas all the region in which we drew is black and where it is black it is cosidered as white,it will create a mask

        imgInv = cv2.cvtColor(imgInv,
                              cv2.COLOR_GRAY2BGR)  # converting again to gray bcoz we have to add in a RGB image i.e img

        # add original img with imgInv ,by doing this we get our drawing only in black color
        img = cv2.bitwise_and(img, imgInv)

        # add img and imgcanvas,by doing this we get colors on img
        img = cv2.bitwise_or(img, imgCanvas)

        # setting the header image
        img[0:125, 0:1280] = header  # on our frame we are setting our JPG image acc to H,W of jpg images

        img[200:300, 100:200] = 255
        cv2.imshow("Image", img)
        # cv2.imshow("Canvas", imgCanvas)
        # cv2.imshow("Inv", imgInv)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    run()


def virtual_sound():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)

    pTime = 0
    # detector = htm.handDetector(detectionCon=0.7)
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # volume.GetMute()
    # volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    vol = 0
    volBar = 400
    volPer = 0
    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9)

    def_hand_length = 0.17
    wCam, hCam = 640, 480
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        success, img = cap.read()
        result = hands.process(img)
        if result.multi_hand_landmarks is not None:

            np_result = settings_vector.get_numpy_points(result.multi_hand_landmarks[0])
            x, y = np_result[8][0], np_result[8][1]
            x_px, y_px = img.shape[1] * x, img.shape[0] * y
            rotate_result = settings_vector.set_standard_position(np_result)
            # x, y = rotate_result[8][0], rotate_result[8][1]
            # x_px, y_px = img.shape[1] * x, img.shape[0] * y
            print(x_px, y_px)
            if 500 < x_px < 600 and 340 < y_px < 440:
                break
            # size_result = settings_vector.set_standard_size(rotate_result)

            # z_o_h_l = def_hand_length / (settings_vector.get_length(rotate_result[0], rotate_result[5]))
            # rotate_result[:, :2] *= z_o_h_l
            # rotate_result[0][2] = z_o_h_l
            # rotate_result[:, 2] += z_o_h_l

            size_result = settings_vector.set_standard_size(rotate_result)
            # settings_vector.draw_hand(size_result, refresh=True)
            length = np.sqrt(np.sum((size_result[8] - size_result[4]) ** 2, axis=0))
            # print(np.sqrt(np.sum((rotate_result[4]-rotate_result[8])**2, axis=0)))
            cv2.circle(img, (int(np_result[4][0] * wCam), int(np_result[4][1] * hCam)), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(np_result[8][0] * wCam), int(np_result[8][1] * hCam)), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (int(np_result[4][0] * wCam), int(np_result[4][1] * hCam)),
                     (int(np_result[8][0] * wCam), int(np_result[8][1] * hCam)), (255, 0, 255), 3)
            cv2.putText(img, f'{np.round(length, 3)}', (wCam - 100, 50), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)
            val_range = [0.2, 1.6]
            vol = np.interp(length, val_range, [minVol, maxVol])
            volBar = np.interp(length, val_range, [400, 150])
            volPer = np.interp(length, val_range, [0, 100])
            volume.SetMasterVolumeLevel(vol, None)
        # lmList = detector.findPosition(img, draw=False)
        # if len(lmList) != 0:
        #     # print(lmList[4], lmList[8])
        #     x1, y1 = lmList[4][1], lmList[4][2]
        #     x2, y2 = lmList[8][1], lmList[8][2]
        #     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        #     cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        #     cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        #     length = math.hypot(x2 - x1, y2 - y1)
        #     # print(length)
        #     # Hand range 50 - 300
        #     # Volume Range -65 - 0
        #     vol = np.interp(length, [0.05, 0.30], [minVol, maxVol])
        #     volBar = np.interp(length, [0.05, 0.30], [400, 150])
        #     volPer = np.interp(length, [0.05, 0.30], [0, 100])
        #     print(int(length), vol)
        #     volume.SetMasterVolumeLevel(vol, None)
        #     if length < 50:
        #         cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        img[340:440, 500:600] = 255
        cv2.imshow("Img", img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    run()


def run():
    test = ClassThreading(speaker_name='baya')
    a = test.start_camera_pipe()


if __name__ == '__main__':
    run()
    # test.start_camera_pipe()

    # test.start_vosk(test.hello_object)

    # test.thread_vosk_hello.start()
    # test.thread_vosk_hello.join()
    # test.what = 'АУЕ БРАТВА'
    # time.sleep(5)
    # test.thread_vosk_hello.start()

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

    def start_vosk(self, what):
        self.thread_vosk_hello = Thread(target=self.vosk_, name=f'thread vosk {self.count_vosk}', args=(what,))
        self.thread_vosk_hello.start()
        self.threads.append(self.thread_vosk_hello)

    def start_kaldi(self, ):
        self.thread_kaldi_ = Thread(target=self.kaldi_, name='thread_kaldi', args=())
        self.thread_kaldi_.start()

    def start_camera_pipe(self):
        self.thread_camera_pipe = Thread(target=self.camera_pipe, name='thread view', args=())
        self.thread_camera_pipe.start()
        self.threads.append(self.thread_camera_pipe)
        self.thread_camera_pipe.join()

    def camera_pipe(self):
        count = 0
        while self.cap.isOpened():
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

    def kaldi_(self, callback=1):

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
                    str_ret = a[f:s]
                    break_ = True
                    return str_ret
                if break_:
                    print('fd')
                    break
                # else:
                #    print(rec.PartialResult())

    def vosk_(self, what):
        while self.cap.isOpened():
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


if __name__ == '__main__':
    test = ClassThreading(speaker_name='baya')
    a = test.start_kaldi()
    # test.start_camera_pipe()

    # test.start_vosk(test.hello_object)

    # test.thread_vosk_hello.start()
    # test.thread_vosk_hello.join()
    # test.what = 'АУЕ БРАТВА'
    # time.sleep(5)
    # test.thread_vosk_hello.start()

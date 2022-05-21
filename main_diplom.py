import cv2
import mediapipe as mp
import pyaudio
from speech_asis import tts, stt
from threading import Thread

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def vosk_(str_: str):
    tts.va_speak(str_)
    thread_kaldi = Thread(target=kaldi_, name='thread_kaldi', args=())
    thread_kaldi.start()


def kaldi_():
    a = stt.va_listen()
    thread_vosk = Thread(target=vosk_, name='thread_vosk',
                         args=(a[1:],))
    thread_vosk.start()
    print(a)

def run():
    cap = cv2.VideoCapture(0)
    count = 0
    # thread_vosk = Thread(target=vosk_, name='thread_vosk', args=(str_))
    # # thread_vosk.start()
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks is not None:
                x, y = results.multi_hand_landmarks[0].landmark[8].x, results.multi_hand_landmarks[0].landmark[8].y
                x_px, y_px = image.shape[1] * x, image.shape[0] * y
                if 500 < x_px < 600 and 340 < y_px < 440:
                    count += 1
                    if count >= 30:
                        print('aer')
                        # tts.va_speak('Ты нажал на кнопку хозяин')
                        count = 0
                        thread_vosk = Thread(target=vosk_, name='thread_vosk',
                                             args=('Кнопка нажата. Теперь я слушаю ваши команды.',))
                        thread_vosk.start()





                else:
                    count = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            image[340:440, 500:600] = 255

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    run()

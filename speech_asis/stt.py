import vosk
import sys
import sounddevice as sd
import queue
import re
import json

model = vosk.Model(r"C:\Users\79614\PycharmProjects\diplom_\speech_asis\model")
samplerate = 16000
device = 1

q = queue.Queue()


def q_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def va_listen(callback=1):
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device, dtype='int16',
                           channels=1, callback=q_callback):

        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            print('говори')
            if rec.AcceptWaveform(data):
                # callback(json.loads(rec.Result())["text"])
                # print(rec.Result())
                a = rec.Result()
                f, s = [m.start() for m in re.finditer('\"', a)][-2:][0], \
                       [m.start() for m in re.finditer('\"', a)][-2:][1]
                str_ret = a[f:s]
                return str_ret
            # else:
            #    print(rec.PartialResult())


# a = va_listen()
# print(a)
import datetime
import pickle

import torch
import sounddevice as sd
import time
import math

from tqdm import tqdm

language = 'ru'
model_id = 'ru_v3'
sample_rate = 48000  # 48000
speaker = 'baya' # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu') # cpu или gpu

model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
model.to(device)


# воспроизводим
def va_speak(what: str):
    audio = model.apply_tts(text=what+"..",
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo)

    sd.play(audio, sample_rate * 1.05)
    time.sleep((len(audio) / sample_rate) + 0.5)
    sd.stop()

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(path_data):
    with open(path_data, 'rb') as inp:
        object_pkl = pickle.load(inp)
    return object_pkl


hello_object = load_object(r'C:\Users\79614\PycharmProjects\diplom_\speach_comand\hello.pkl')
sd.play(hello_object, sample_rate * 1.05)

start = datetime.datetime.now()
# va_speak('Приветствую хозяин. Я готова слушать Ваши команды.')
time_ = datetime.datetime.now() - start
a = 0
# str_ = ''
# with open("ya_idu_k_reke.txt", "r", encoding='utf8') as fh:
#     a = fh.readlines()
#     for i in a:
#         str_ += i
#     a = 0
# str_ = str_.replace('\n', ' ')
#
# a = 0
# # va_speak(str_)
# a_ = str_[0]
# str_.replace(a_, '')
# list_str = str_.split(' ')
# for i in tqdm(range(math.ceil(len(list_str)/50))):
#     start_index = i*50
#     va_speak(" ".join(list_str[start_index:start_index+50]))

# sd.play(audio, sample_rate)
# time.sleep(len(audio) / sample_rate)
# sd.stop()
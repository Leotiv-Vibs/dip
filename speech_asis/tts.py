import torch
import sounddevice as sd
import time
import math

from tqdm import tqdm

language = 'ru'
model_id = 'ru_v3'
sample_rate = 24000 # 48000
speaker = 'baya' # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu') # cpu или gpu
text = "Хауди Хо, друзья!!!"

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
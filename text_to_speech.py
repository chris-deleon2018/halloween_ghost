from gtts import gTTS
from playsound import playsound
import librosa
import soundfile


language = 'en'
myobj = gTTS(text='Hey! Little kid! Take 3 pieces of candy.', lang=language, slow=False)
myobj.save("speak.mp3")

#playsound("speak.mp3")

y, sr = librosa.load("speak.mp3")
new_y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)
new_y = librosa.effects.time_stretch(new_y, rate=0.9)
soundfile.write("speak.wav", new_y, sr,)

ghost_y, ghost_sr = librosa.load("ghost_moan.mp3")
#ghost_y = librosa.effects.pitch_shift(ghost_y, sr=ghost_sr, n_steps=-6)
ghost_y = librosa.effects.time_stretch(ghost_y, rate=1.5)
soundfile.write("ghost_moan.wav", ghost_y, sr,)

playsound("ghost_moan.wav")
playsound("speak.wav")
#playsound("Ghosts Electronic Simulation - QuickSounds.com.mp3")
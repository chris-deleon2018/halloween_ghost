import cv2 as cv
from gtts import gTTS
from playsound import playsound
import time
import librosa
import soundfile
import re

net = cv.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

language = 'en'

# Get Class ID Names
classes = []
with open("classes.txt", "r") as f:
    for class_name in f.read().splitlines():
        classes.append(class_name)

phrases = [
    "Hello <object_name>! I am going to eat you!",
    "O hey look everyone! It's a <object_name>!"
]
phrase_count = 0
wait_count = 0
wait_count_threshold = 50
object_name = ""
objects = set()

while True:
    ret, frame = cap.read()
    
    (class_ids, scores, bboxes) = model.detect(frame)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        if score >= 0.80:
            object_name = classes[class_id]
            (x,y,w,h) = bbox
            cv.rectangle(frame,(x, y),(x + w, y + h),(200,0,50),3)
            cv.putText(frame, f'{object_name}: {round(score*100,2)}', (x, y-5), cv.FONT_HERSHEY_PLAIN, 1, (200,0,50), 2 )
            objects.add(object_name)
    if wait_count == wait_count_threshold and not len(objects) == 0:
        nouns = ""
        #print(f'objects: {objects}')
        if len(objects) == 2:
            nouns = "<obj> and <obj>"
            for obj in objects:
                nouns = nouns.replace("<obj>",obj,1)
            #print(nouns)
            objects.clear()
        elif len(objects) >= 3:
            count = 0
            total_count = len(objects)
            for obj in objects:
                if count == total_count-1:
                    nouns += f'and {obj}'
                else:
                    nouns += f'{obj}, '
                count += 1
            #print(nouns)
            objects.clear()
        else:
            nouns = objects.pop()
        phrase = re.sub(r'<object_name>', nouns, phrases[phrase_count % 2])
        print(phrase) 
        myobj = gTTS(text=phrase, lang=language, slow=False)
        print(phrase_count)
        print(phrase)
        myobj.save("speak.mp3")
        y, sr = librosa.load("speak.mp3")
        new_y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)
        new_y = librosa.effects.time_stretch(new_y, rate=0.9)
        soundfile.write("speak.wav", new_y, sr,)
        playsound("ghost_moan.wav")
        playsound("speak.wav")
        wait_count = 0
        object_name = ""
        phrase_count += 1
    wait_count += 1
            
        #TODO Add logic to determine how many people
        
    cv.imshow("Frame",frame)
    cv.waitKey(1)
    #time.sleep(2)
# Halloween Ghost - Person Detection

This was a fun Halloween project that requires a speaker (preferrably a wireless bluetooth speaker), a web camera connected to the computer, and a halloween ghost prop. The idea is to have a bowl of candy with a ghost halloween prop with a speaker hiding under the ghost to detect trick or treaters to greet the visitors with a set of phrases. These phrases are customizable and will be translated by the text to speech library. The object detection is a trained neural net provided by YoloV4, and being used to scrape the video feed to look for a person. If a person is detected, the phrase will be said in a rotating order and a video frame (picture) will be saved to capture the visitors. Disclaimer - Please check your local rules and regulations/laws related to video camera recordings. 

# Dependancies required

```
python3 -m pip install opencv-python
python3 -m pip install gTTS
python3 -m pip install gtts playsound
python3 -m pip install PyObjC
python3 -m pip install librosa
python3 -m pip install regex
```

# How to run
`python3 main_detect.py`

import speech_recognition as sr

def transcribe_wavefile(speech, language):
    '''
    Use sr.Recognizer.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.
    '''
    recognizer = sr.Recognizer()

    with sr.AudioFile(speech) as source:
        audio = recognizer.record(source)

    text = recognizer.recognize_google(audio, language=language)

    return text
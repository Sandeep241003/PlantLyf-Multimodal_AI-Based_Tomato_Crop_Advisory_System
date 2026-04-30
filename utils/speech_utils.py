import speech_recognition as sr

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except Exception as e:
        return None
import whisper
import tempfile
from pydub import AudioSegment

model = whisper.load_model("base")

def speech_to_text_whisper(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_bytes)
            raw_path = tmp.name

        sound = AudioSegment.from_file(raw_path)
        sound = sound.set_channels(1).set_frame_rate(16000)

        converted_path = raw_path.replace(".webm", ".wav")
        sound.export(converted_path, format="wav")

        result = model.transcribe(converted_path)

        return result["text"]

    except Exception as e:
        print("Whisper error:", e)
        return None
import re
import os
import io
import time
import tempfile
import asyncio
import threading

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from groq import Groq
import edge_tts

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from streamlit_geolocation import streamlit_geolocation
from weather import get_all_realtime_factors
from disease_model import analyze_uploaded_image
from utils.language_utils import detect_language, translate_to_english, translate_from_english

# ---------------- LOAD ENV ----------------
load_dotenv()
DB_FAISS_PATH = "vectorstore/db_faiss"

if not os.getenv("GROQ_API_KEY"):
    st.error("Missing `GROQ_API_KEY`. Add it to Streamlit Cloud Secrets (or local .env).")
    st.stop()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- VECTORSTORE ----------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# ---------------- HELPERS ----------------
def remove_large_headings(text: str) -> str:
    return re.sub(r'^#{1,6}\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)

def has_valid_realtime_data(realtime):
    if not realtime:
        return False
    important_fields = [
        "temperature", "humidity", "wind_speed",
        "soil_moisture", "soil_temperature",
        "rain_1h", "evapotranspiration"
    ]
    return any(realtime.get(k) is not None for k in important_fields)

def transcribe_audio(audio_bytes):
    print("Audio size:", len(audio_bytes), "bytes")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        with open(tmp_path, "rb") as audio_file:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=("audio.wav", audio_file, "audio/wav"),
                # NO language param = Groq auto-detects
                # task="transcribe" = keeps original language (Hindi stays Hindi)
            )

        text = result.text.strip()
        print("Transcribed:", text)

        hallucinations = {"you", "thank you", "thanks", "thanks for watching", "bye", ""}
        if text.lower() in hallucinations:
            print("Hallucination detected:", text)
            return ""

        return text

    except Exception as e:
        print("transcribe_audio error:", e)
        return ""

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def generate_audio_file(text: str) -> str:
    """Generate TTS audio and return the temp file path. Fully synchronous."""
    async def _gen():
        voice = "en-IN-NeerjaNeural"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            temp_path = f.name
        communicate = edge_tts.Communicate(text[:], voice)
        await communicate.save(temp_path)
        return temp_path

    # Use a fresh event loop — safe in Streamlit's environment
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_gen())
    finally:
        loop.close()

def speak(text: str):
    """Generate and immediately play TTS audio inline."""
    try:
        audio_path = generate_audio_file(text)
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, autoplay=True)
            # Clean up after rendering
            try:
                os.remove(audio_path)
            except Exception:
                pass
    except Exception as e:
        print("TTS error:", e)

# ---------------- BUILD CHAIN ----------------
def build_chain(vectorstore, realtime):
    use_realtime = has_valid_realtime_data(realtime)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    if use_realtime:
        realtime_section = f"""
Location: {realtime.get("location")}
Temperature: {realtime.get("temperature")} °C
Humidity: {realtime.get("humidity")} %
Wind Speed: {realtime.get("wind_speed")} m/s
Rainfall: {realtime.get("rain_1h")} mm
Cloud Cover: {realtime.get("cloud_cover")} %
Soil Moisture: {realtime.get("soil_moisture")}
Soil Temperature: {realtime.get("soil_temperature")} °C
Evapotranspiration: {realtime.get("evapotranspiration")}
"""
    else:
        realtime_section = "Environmental data unavailable."

    template = """
You are a professional agricultural advisor specializing in tomato cultivation.

You MUST integrate real-time environmental data naturally.

Guidelines:
- use bullet points for steps if required
- give practical solutions

If question is unrelated → respond:
"I'm specialized in tomato plant care and can only assist with tomato-related queries."

-------------------------
ENVIRONMENT
-------------------------
{realtime_section}

-------------------------
CONTEXT
-------------------------
{context}

-------------------------
QUESTION
-------------------------
{question}

-------------------------
ANSWER
-------------------------
"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt.partial(realtime_section=realtime_section)
        | ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=800,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        | StrOutputParser()
    )

    return rag_chain

# ---------------- MAIN APP ----------------
def main():
    st.markdown("""
    <style>
    .responsive-title {
        font-size: clamp(22px, 4vw, 40px);
        font-weight: 700;
        text-align: center;
        line-height: 1.2;
        word-break: break-word;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="responsive-title">🍅 PlantLyf: Tomato Plant Care Chatbot</div>', unsafe_allow_html=True)

    # -------- SESSION STATE --------
    defaults = {
        "messages": [],
        "realtime": None,
        "weather_loaded": False,
        "disease_result": None,
        "uploaded_file_meta": None,
        "ignore_uploaded_image": False,
        "image_uploader_key": 0,
        "last_audio_key": None,   # ← track processed audio to avoid re-processing
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # -------- SIDEBAR --------
    st.sidebar.header("📍 Location")
    gps_location = streamlit_geolocation()
    city = st.sidebar.selectbox(
        "Fallback city",
        ["Prayagraj", "Lucknow", "Varanasi", "Agra", "Kanpur", "Ghaziabad",
         "Noida", "Meerut", "Saharanpur", "Gorakhpur", "Jhansi", "Mathura",
         "Ayodhya", "Chitrakoot"]
    )

    # -------- FETCH WEATHER --------
    if not st.session_state.weather_loaded:
        with st.spinner("Fetching weather data..."):
            if gps_location and gps_location.get("latitude"):
                st.session_state.realtime = get_all_realtime_factors(
                    lat=gps_location["latitude"],
                    lon=gps_location["longitude"]
                )
            else:
                st.session_state.realtime = get_all_realtime_factors(location_name=city)
        st.session_state.weather_loaded = True

    realtime = st.session_state.realtime

    if realtime:
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.markdown(f"**📍 {realtime.get('location')}**")
        with col2:
            if st.button("🔄"):
                st.session_state.weather_loaded = False
                st.rerun()

    # -------- IMAGE UPLOAD --------
    st.sidebar.header("🍃 Tomato leaf image")
    uploaded_file = st.sidebar.file_uploader(
        "Upload leaf image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"leaf_uploader_{st.session_state.image_uploader_key}",
    )

    if uploaded_file is not None:
        current_meta = (uploaded_file.name, getattr(uploaded_file, "size", None))
        if current_meta != st.session_state.uploaded_file_meta:
            st.session_state.uploaded_file_meta = current_meta
            st.session_state.ignore_uploaded_image = False

    if uploaded_file is not None and not st.session_state.ignore_uploaded_image:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.sidebar.image(image, caption="Uploaded leaf image", use_container_width=True)
            disease_result = analyze_uploaded_image(uploaded_file)
            st.session_state.disease_result = disease_result
        except Exception as e:
            st.sidebar.warning(f"Could not analyze image: {e}")
            st.session_state.disease_result = None

    disease_result = st.session_state.disease_result

    if disease_result:
        st.sidebar.markdown("**Image-based disease analysis**")
        st.sidebar.write(
            f"- Predicted: `{disease_result['predicted_label']}` "
            f"(confidence: {disease_result['confidence']:.2f})"
        )
        if st.sidebar.button("Clear image"):
            st.session_state.disease_result = None
            st.session_state.ignore_uploaded_image = True
            st.session_state.uploaded_file_meta = None
            disease_result = None
            st.session_state.image_uploader_key += 1
            st.rerun()

    # -------- CHAT HISTORY --------
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # -------- BOTTOM INPUT BAR --------
    st.markdown("""
    <style>

    /* Give space so content is not hidden */
    .main .block-container { padding-bottom: 100px; }

    /* ===== MIC BUTTON ===== */
    div[data-testid="stAudioInput"] {
        position: fixed;
        bottom: 18px;
        right: 30px;   /* 👈 FIXED (was 80px) */
        z-index: 99999;
        width: auto !important;
    }

    /* Hide labels */
    div[data-testid="stAudioInput"] label,
    div[data-testid="stAudioInput"] small {
        display: none !important;
    }

    /* Remove wrapper background */
    div[data-testid="stAudioInput"] > div {
        background: transparent !important;
    }

    /* Actual mic button */
    div[data-testid="stAudioInput"] button {
        background: #ff4b4b !important;
        border-radius: 50% !important;
        width: 52px !important;
        height: 52px !important;
        border: none !important;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(255,75,75,0.4);
        transition: all 0.2s ease;
    }

    /* Icon */
    div[data-testid="stAudioInput"] button svg {
        fill: white !important;
        width: 22px;
        height: 22px;
    }

    /* Remove blue ring */
    div[data-testid="stAudioInput"] button:focus,
    div[data-testid="stAudioInput"] button:active {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Hover */
    div[data-testid="stAudioInput"] button:hover {
        background: #e63939 !important;
        transform: scale(1.05);
    } 

    /* ===== CHAT INPUT ===== */
    div[data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 15px !important;
        left: 350px !important;
        right: 220px !important;  /* 👈 leave space for mic */
        z-index: 99998 !important;
    }

    /* ===== BOTTOM BAR ===== */
    section[data-testid="stMain"] > div:last-child {
        position: fixed;
        bottom: 0;
        left: 300px;
        right: 0;
        height: 75px;
        background: white;
        border-top: 1px solid #e0e0e0;
        z-index: 99990;
        display: flex;
        align-items: center;
    }

    </style>
    """, unsafe_allow_html=True)

    user_query = st.chat_input("Ask in any language...")
    audio_input = st.audio_input("mic", label_visibility="collapsed")

    voice_text = None

    if audio_input is not None:
        # Use the file ID to detect if this is a NEW recording (avoid reprocessing on reruns)
        audio_key = getattr(audio_input, "file_id", id(audio_input))
        if audio_key != st.session_state.last_audio_key:
            st.session_state.last_audio_key = audio_key
            audio_bytes = audio_input.read()
            print("Audio size:", len(audio_bytes), "bytes")

            if len(audio_bytes) < 5000:
                st.warning("Too short, please try again.")
            else:
                with st.spinner("Transcribing..."):
                    voice_text = transcribe_audio(audio_bytes)
                if voice_text:
                    st.success(f"You said: {voice_text}")
                else:
                    st.warning("Could not understand. Please speak clearly.")

    if voice_text:
        user_query = voice_text

    if user_query:
        st.chat_message("user").markdown(user_query)

        lang = detect_language(user_query)
        query_en = translate_to_english(user_query, lang)

        if st.session_state.disease_result:
            query_en += f"\nDetected disease: {st.session_state.disease_result['predicted_label']}"

        try:
            vectorstore = get_vectorstore()
            rag_chain = build_chain(vectorstore, realtime)

            answer_en = rag_chain.invoke(query_en)
            answer_en = remove_large_headings(answer_en)

            final_answer = translate_from_english(answer_en, lang)

            st.chat_message("assistant").markdown(final_answer)

            # ✅ Synchronous TTS — no threading, no sleep, no race condition
            speak(final_answer)

            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
import os
import sys
import queue
import sounddevice as sd
import vosk
import json
import time
import pyttsx3
import langid
import streamlit as st
import requests

# Function: Remove Duplicates from Text
def remove_duplicates(text):
    words = text.split()
    deduplicated = []
    for word in words:
        if not deduplicated or word != deduplicated[-1]:
            deduplicated.append(word)
    return ' '.join(deduplicated)

# Initialize Vosk Model
model_path = "./vosk-model-small-en-us-0.15"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model Vosk tidak ditemukan di path: {model_path}")
model = vosk.Model(model_path)

samplerate = 16000
device = None
audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(bytes(indata))

def get_speech_to_text():
    silence_timeout = 5
    last_spoken_time = time.time()
    collected_text = ""

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device,
                           dtype='int16', channels=1, callback=audio_callback):
        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            try:
                data = audio_q.get(timeout=silence_timeout)
                last_spoken_time = time.time()

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    final_text = result.get("text", "").strip()
                    final_text = remove_duplicates(final_text)
                    return final_text

            except queue.Empty:
                if time.time() - last_spoken_time > silence_timeout:
                    return collected_text.strip()

# Function: Text-to-Speech
def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    language, _ = langid.classify(text)

    if language == 'id':
        engine.setProperty('voice', voices[0].id)
    elif language == 'en':
        engine.setProperty('voice', voices[1].id)
    else:
        engine.setProperty('voice', voices[1].id)

    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# Streamlit Interface
# API_URL = "http://34.101.169.131:8000"
API_URL = "http://34.101.254.104:8000"

st.title("Chatbot Multimodal (Teks & Suara)")

input_mode = st.radio("Pilih metode input:", ("Teks", "Suara"))

if input_mode == "Teks":
    user_text = st.text_input("Masukkan teks:")
    if st.button("Kirim"):
        if user_text.strip():
            response = requests.post(f"{API_URL}/chat", json={"prompt": user_text, "max_length": 100})
            if response.status_code == 200:
                bot_response = response.json()["response"]
                st.text_area("Respons Chatbot:", bot_response, height=150)
                speak(bot_response)  # TTS untuk respons chatbot
            else:
                st.error("Gagal menghubungi server.")
        else:
            st.warning("Teks tidak boleh kosong!")

elif input_mode == "Suara":
    if st.button("Mulai Rekaman"):
        st.info("Sedang merekam suara...")
        try:
            input_text = get_speech_to_text()
            st.write(f"Input Suara ke Teks: {input_text}")

            response = requests.post(f"{API_URL}/chat", json={"prompt": input_text, "max_length": 100})
            if response.status_code == 200:
                bot_response = response.json()["response"]
                st.text_area("Respons Chatbot:", bot_response, height=150)
                speak(bot_response)  # TTS untuk respons chatbot
            else:
                st.error("Gagal menghubungi server.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

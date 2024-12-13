from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import json
import re
import random
from datetime import datetime
import torch

# Inisialisasi model LLaMA.cpp
model_path = "./bitnet_b1_58-large.Q4_0.gguf"  # Ganti dengan path model Anda
try:
    model = Llama(model_path=model_path)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    raise RuntimeError("Gagal memuat model.")

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Menambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan akses dari semua sumber
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode HTTP
    allow_headers=["*"],  # Izinkan semua header
)

# File untuk menyimpan riwayat chat
CHAT_HISTORY_FILE = "chat_history.json"

# Fungsi untuk memuat riwayat chat
def load_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []  # Jika file tidak ditemukan, kembalikan daftar kosong
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

# Fungsi untuk menyimpan riwayat chat
def save_chat_history(chat_data):
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump(chat_data, file, indent=4)
    except Exception as e:
        print(f"Error saving chat history: {e}")

# Fungsi deteksi bahasa
def detect_language(text):
    indonesian_vocab = [
        "apa", "itu", "adalah", "sebuah", "di", "yang", "dari", "ke", "dan", "saya", "kami", "mereka", "dia", "akan",
        "pernah", "baru", "untuk", "mungkin", "lebih", "tidak", "boleh", "ada", "belum", "sekarang", "harus", "ini",
        "karena", "lagi", "seperti", "besar", "kecil", "semua", "dengan", "tersebut", "sama", "masih", "langsung",
        "tapi", "saat", "tunggu", "selalu", "atau", "pada", "memiliki", "cuma", "mengapa", "berapa", "baik", "juga",
        "hanya", "mau", "sangat", "mencoba"
    ]
    chunks = re.split(r'[.?!,;]', text)
    for chunk in chunks:
        words = chunk.split()
        vocab_match_count = sum(1 for word in words if word in indonesian_vocab)
        if vocab_match_count >= 2:
            return 'id'
    return detect(text)

# Fungsi terjemahan
def translate(text, src_lang, tgt_lang):
    model_path = f"./local_models/opus-mt-{src_lang}-{tgt_lang}_model"
    tokenizer_path = f"./local_models/opus-mt-{src_lang}-{tgt_lang}_tokenizer"
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)

    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translation = model.generate(**input_ids)
    return tokenizer.decode(translation[0], skip_special_tokens=True)

# Model untuk request data
class ChatRequest(BaseModel):
    prompt: str
    max_length: int = 100  # Panjang maksimum teks yang dihasilkan

# Endpoint utama untuk chatbot
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Mengambil data dari request
        prompt = request.prompt.strip()
        max_length = request.max_length

        # Validasi input
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
        if max_length > 512 or max_length < 10:
            raise HTTPException(
                status_code=400, detail="max_length must be between 10 and 512."
            )

        # Deteksi bahasa
        language = detect_language(prompt)
        prompt_with_instruction_en = f"Provide a concise and direct answer to the following question: {prompt}"
        prompt_with_instruction_id = f"Berikan jawaban yang ringkas dan langsung untuk pertanyaan berikut: {prompt}"

        # Terjemahan jika bahasa adalah Indonesia
        if language == 'id':
            translated_prompt = translate(prompt_with_instruction_id, 'id', 'en')
            response = model(
                prompt=translated_prompt,
                max_tokens=max_length,
                temperature=random.uniform(0.5, 1.0),
                top_p=random.uniform(0.8, 1.0),
                top_k=40
            )
            generated_text_en = response["choices"][0]["text"].strip()
            generated_text = translate(generated_text_en, 'en', 'id')
        else:
            response = model(
                prompt=prompt_with_instruction_en,
                max_tokens=max_length,
                temperature=random.uniform(0.5, 1.0),
                top_p=random.uniform(0.8, 1.0),
                top_k=40
            )
            generated_text = response["choices"][0]["text"].strip()

        # Simpan riwayat chat
        chat_history = load_chat_history()
        chat_history.append({
            "user_message": prompt,
            "bot_response": generated_text,
            "timestamp": datetime.utcnow().isoformat()
        })
        save_chat_history(chat_history)

        return {"response": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# Endpoint untuk mendapatkan riwayat chat
@app.get("/history")
async def get_history():
    try:
        chat_history = load_chat_history()
        return {"history": chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error loading chat history.")

# Endpoint untuk tes apakah API berjalan
@app.get("/")
async def root():
    return {"message": "Chatbot API is running with LLaMA.cpp and translation support."}

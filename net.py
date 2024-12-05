from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import json
from datetime import datetime

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

        # Menghasilkan respons teks
        try:
            response = model(
                prompt=prompt,
                max_tokens=max_length,
                temperature=0.7,  # Penyesuaian suhu untuk kontrol kreativitas
                top_p=0.95,  # Top-p sampling
                repeat_penalty=1.2,  # Penalti pengulangan
            )["choices"][0]["text"]
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating response: {e}"
            )

        # Simpan riwayat chat
        chat_history = load_chat_history()
        chat_history.append({
            "user_message": prompt,
            "bot_response": response.strip(),
            "timestamp": datetime.utcnow().isoformat()
        })
        save_chat_history(chat_history)

        return {"response": response.strip()}

    except Exception as e:
        # Batasi informasi error untuk keamanan
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

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
    return {"message": "Chatbot API is running with LLaMA.cpp"}

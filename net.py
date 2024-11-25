from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama

# Inisialisasi model LLaMA.cpp
model_path = "./bitnet_b1_58-large.Q8_0.gguf"  # Ganti dengan path model Anda
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

        return {"response": response.strip()}

    except Exception as e:
        # Batasi informasi error untuk keamanan
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# Endpoint untuk tes apakah API berjalan
@app.get("/")
async def root():
    return {"message": "Chatbot API is running with LLaMA.cpp"}

import os
from dotenv import load_dotenv
import uuid
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
from groq import Groq
import json
from datetime import datetime
import sqlite3
from typing import List, Optional

app = FastAPI()

# Statik dosyaları (HTML, ses dosyaları vb.) sunmak için /static yolunu bağlar
app.mount("/static", StaticFiles(directory="static"), name="static")

# .env dosyasını yükle
load_dotenv()

# Ortam değişkenlerinden API anahtarlarını ve ses ID'sini al
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# Değişkenlerin yüklendiğini kontrol et
print("GROQ_API_KEY:", "******" if GROQ_API_KEY else "Not Set")
print("ELEVENLABS_API_KEY:", "******" if ELEVENLABS_API_KEY else "Not Set")
print("ELEVENLABS_VOICE_ID:", ELEVENLABS_VOICE_ID if ELEVENLABS_VOICE_ID else "Not Set")
print("BRAVE_API_KEY:", "******" if BRAVE_API_KEY else "Not Set")

if not GROQ_API_KEY or not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
    print("Warning: One or more environment variables are missing. Check .env file or system environment variables.")

# API URL'leri
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Groq istemcisini başlat
client = Groq(api_key=GROQ_API_KEY)

# Veritabanı başlatma
def init_database():
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_question TEXT NOT NULL,
            ai_answer TEXT NOT NULL,
            personality TEXT NOT NULL,
            audio_url TEXT,
            session_id TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Veritabanını başlat
init_database()

# Konuşma geçmişi yönetimi
class ConversationManager:
    def __init__(self):
        self.db_path = 'conversations.db'
    
    def save_conversation(self, user_question: str, ai_answer: str, personality: str, audio_url: str = None, session_id: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (timestamp, user_question, ai_answer, personality, audio_url, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), user_question, ai_answer, personality, audio_url, session_id))
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, limit: int = 50, session_id: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (session_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        conversations = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": conv[0],
                "timestamp": conv[1],
                "user_question": conv[2],
                "ai_answer": conv[3],
                "personality": conv[4],
                "audio_url": conv[5],
                "session_id": conv[6]
            }
            for conv in conversations
        ]
    
    def delete_conversation(self, conversation_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        conn.commit()
        conn.close()
    
    def clear_history(self, session_id: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if session_id:
            cursor.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
        else:
            cursor.execute('DELETE FROM conversations')
        conn.commit()
        conn.close()

# Konuşma yöneticisi instance'ı
conversation_manager = ConversationManager()

# Personality sistemleri
PERSONALITIES = {
    "engineer": {
        "system_prompt": """Sen deneyimli bir yazılım mühendisisin. Teknik konularda detaylı ve pratik çözümler sunarsın. 
        Kod örnekleri verir, performans optimizasyonları önerir ve en iyi pratikleri paylaşırsın. 
        Türkçe konuş ve teknik terimleri açıkla."""
    },
    "scientist": {
        "system_prompt": """Sen meraklı bir bilim insanısın. Her konuyu bilimsel açıdan analiz eder, 
        araştırma yöntemlerini açıklar ve kanıta dayalı yaklaşımlar önerirsin. 
        Karmaşık kavramları basit terimlerle açıklarsın. Türkçe konuş."""
    },
    "planner": {
        "system_prompt": """Sen stratejik bir planlayıcısın. Her sorunu adım adım analiz eder, 
        organize çözümler sunar ve uzun vadeli planlar önerirsin. 
        Proje yönetimi, zaman yönetimi ve hedef belirleme konularında uzmansın. Türkçe konuş."""
    }
}

# Varsayılan personality
current_personality = "engineer"

# Ses kalitesi ayarları
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True
}

# İstek gövdesi için Pydantic modelleri
class QuestionRequest(BaseModel):
    question: str

class PersonalityRequest(BaseModel):
    personality: str

class VoiceSettingsRequest(BaseModel):
    stability: Optional[float] = 0.5
    similarity_boost: Optional[float] = 0.75
    style: Optional[float] = 0.0
    use_speaker_boost: Optional[bool] = True

class SessionRequest(BaseModel):
    session_id: str

# Web search fonksiyonu
async def web_search(query: str, count: int = 5):
    """Brave Search API kullanarak web araması yapar"""
    if not BRAVE_API_KEY:
        return None
    
    try:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        
        params = {
            "q": query,
            "count": count,
            "search_lang": "tr_TR"
        }
        
        response = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "web" in data and "results" in data["web"]:
            for result in data["web"]["results"][:count]:
                results.append({
                    "title": result.get("title", ""),
                    "description": result.get("description", ""),
                    "url": result.get("url", "")
                })
        
        return results
    except Exception as e:
        print(f"Web search error: {e}")
        return None

# Gürültü filtreleme fonksiyonu
def apply_noise_filter(audio_data: bytes) -> bytes:
    """Basit gürültü filtreleme - gerçek uygulamada daha gelişmiş algoritmalar kullanılabilir"""
    # Bu basit bir örnek - gerçek uygulamada librosa veya pydub kullanılabilir
    return audio_data

# Personality değiştirme endpoint'i
@app.post("/set-personality")
async def set_personality(request: PersonalityRequest):
    global current_personality
    personality = request.personality
    
    if personality not in PERSONALITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geçersiz personality: {personality}. Geçerli seçenekler: {list(PERSONALITIES.keys())}"
        )
    
    current_personality = personality
    return JSONResponse({"message": f"Personality {personality} olarak ayarlandı"})

# Ses ayarları endpoint'i
@app.post("/set-voice-settings")
async def set_voice_settings(request: VoiceSettingsRequest):
    global DEFAULT_VOICE_SETTINGS
    DEFAULT_VOICE_SETTINGS.update(request.dict(exclude_unset=True))
    return JSONResponse({"message": "Ses ayarları güncellendi", "settings": DEFAULT_VOICE_SETTINGS})

# Konuşma geçmişi endpoint'leri
@app.get("/conversation-history")
async def get_conversation_history(limit: int = 50, session_id: Optional[str] = None):
    history = conversation_manager.get_conversation_history(limit, session_id)
    return JSONResponse({"conversations": history})

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: int):
    conversation_manager.delete_conversation(conversation_id)
    return JSONResponse({"message": "Konuşma silindi"})

@app.delete("/clear-history")
async def clear_history(session_id: Optional[str] = None):
    conversation_manager.clear_history(session_id)
    return JSONResponse({"message": "Geçmiş temizlendi"})

@app.post("/ask")
async def ask(request_data: QuestionRequest):
    question = request_data.question

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Groq API Key (GROQ_API_KEY) ortam değişkeni ayarlı değil."
        )
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ElevenLabs API Key (ELEVENLABS_API_KEY) ortam değişkeni ayarlı değil."
        )
    if not ELEVENLABS_VOICE_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ElevenLabs Voice ID (ELEVENLABS_VOICE_ID) ortam değişkeni ayarlı değil."
        )

    # Dil algılama için varsayılan olarak Türkçe
    detected_language = "tr"
    if any(word.lower() in question.lower() for word in ["hello", "hi", "yes", "no"]):
        detected_language = "en"

    # Web search yap (eğer güncel bilgi gerekiyorsa)
    web_results = None
    if any(keyword in question.lower() for keyword in ["güncel", "son", "haber", "bugün", "recent", "latest", "news"]):
        web_results = await web_search(question)

    # --- Groq API'sine İstek Gönder ---
    try:
        groq_headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Personality'ye göre sistem prompt'u seç
        system_prompt = PERSONALITIES[current_personality]["system_prompt"]
        
        # Web search sonuçlarını ekle
        if web_results:
            context = "Güncel web arama sonuçları:\n"
            for i, result in enumerate(web_results, 1):
                context += f"{i}. {result['title']}\n{result['description']}\nURL: {result['url']}\n\n"
            
            user_content = f"Kullanıcı sorusu: {question}\n\n{context}\nBu bilgileri kullanarak cevap ver."
        else:
            user_content = question
        
        groq_data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 700
        }
        print("\n[DEBUG] Groq Request Headers:", groq_headers)
        print("[DEBUG] Groq Request Data:", groq_data)

        groq_response = requests.post(GROQ_URL, headers=groq_headers, json=groq_data, timeout=90)
        groq_response.raise_for_status()

        groq_response_json = groq_response.json()
        if not groq_response_json.get("choices") or not groq_response_json["choices"][0].get("message") or not groq_response_json["choices"][0]["message"].get("content"):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Groq API returned an unexpected response format or empty content.")
        answer = groq_response_json["choices"][0]["message"]["content"]
        print(f"[DEBUG] Groq Response (Answer): {answer[:100]}...")

    except requests.exceptions.RequestException as e:
        error_detail_groq = e.response.text if e.response is not None else str(e)
        print(f"[ERROR] Groq API Request Error: {error_detail_groq}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Groq API bağlantı/istek hatası: {error_detail_groq}"
        )
    except (KeyError, IndexError) as e:
        print(f"[ERROR] Groq API Response Parsing Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Groq API yanıt formatı beklenenden farklı veya eksik: {e}"
        )
    except Exception as e:
        print(f"[ERROR] Unexpected Groq Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Groq API ile ilgili beklenmedik hata: {e}")

    # --- ElevenLabs API'sine İstek Gönder ---
    try:
        elevenlabs_url = f"{ELEVENLABS_BASE_URL}/{ELEVENLABS_VOICE_ID}"
        elevenlabs_headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        elevenlabs_data = {
            "text": answer,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": DEFAULT_VOICE_SETTINGS
        }
        print("\n[DEBUG] ElevenLabs Request Headers:", elevenlabs_headers)
        print("[DEBUG] ElevenLabs Request Data:", elevenlabs_data)

        elevenlabs_response = requests.post(elevenlabs_url, headers=elevenlabs_headers, json=elevenlabs_data, timeout=120)
        elevenlabs_response.raise_for_status()

        audio_data = elevenlabs_response.content
        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ElevenLabs API'den boş ses verisi alındı."
            )
        print(f"[DEBUG] ElevenLabs Audio Data Size: {len(audio_data)} bytes")

    except requests.exceptions.RequestException as e:
        error_detail_eleven = e.response.text if e.response is not None else str(e)
        print(f"[ERROR] ElevenLabs API Request Error: {error_detail_eleven}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ElevenLabs API bağlantı/istek hatası: {e} - Detay: {error_detail_eleven}"
        )
    except Exception as e:
        print(f"[ERROR] Unexpected ElevenLabs Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"ElevenLabs API ile ilgili beklenmedik hata: {e}")

    # --- Ses Dosyasını Kaydet ---
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = os.path.join("static", audio_filename)
    
    try:
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        print(f"[DEBUG] Audio file saved to: {audio_path}")
    except IOError as e:
        print(f"[ERROR] File Save Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ses dosyası kaydedilirken hata oluştu: {e}"
        )
        
    audio_url = f"/static/{audio_filename}"
    print(f"[DEBUG] Audio URL for frontend: {audio_url}")

    # Konuşma geçmişine kaydet
    conversation_manager.save_conversation(question, answer, current_personality, audio_url)

    return JSONResponse({"answer": answer, "audio_url": audio_url})

# Yeni bir endpoint: Ses dosyasını yükleyip Groq Whisper ile metne çevirme
@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Groq API Key (GROQ_API_KEY) ortam değişkeni ayarlı değil."
        )
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ElevenLabs API Key (ELEVENLABS_API_KEY) ortam değişkeni ayarlı değil."
        )
    if not ELEVENLABS_VOICE_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ElevenLabs Voice ID (ELEVENLABS_VOICE_ID) ortam değişkeni ayarlı değil."
        )

    try:
        # Ses dosyasını oku
        audio_content = await file.read()
        
        # Gürültü filtreleme uygula
        filtered_audio = apply_noise_filter(audio_content)
        
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join("static", audio_filename)
        with open(audio_path, "wb") as f:
            f.write(filtered_audio)

        # Groq Whisper ile sesi metne çevir
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        print(f"[DEBUG] Whisper Transcription: {transcription}")

        # Dil algılama (basit tahmin)
        detected_language = "tr"  # Varsayılan Türkçe
        if any(word.lower() in transcription.lower() for word in ["hello", "hi", "yes", "no"]):
            detected_language = "en"

        # Web search yap (eğer güncel bilgi gerekiyorsa)
        web_results = None
        if any(keyword in transcription.lower() for keyword in ["güncel", "son", "haber", "bugün", "recent", "latest", "news"]):
            web_results = await web_search(transcription)

        # Çeviriyi Groq'a gönder (dil tahminiyle)
        groq_headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Personality'ye göre sistem prompt'u seç
        system_prompt = PERSONALITIES[current_personality]["system_prompt"]
        
        # Web search sonuçlarını ekle
        if web_results:
            context = "Güncel web arama sonuçları:\n"
            for i, result in enumerate(web_results, 1):
                context += f"{i}. {result['title']}\n{result['description']}\nURL: {result['url']}\n\n"
            
            user_content = f"Kullanıcı sorusu: {transcription}\n\n{context}\nBu bilgileri kullanarak cevap ver."
        else:
            user_content = transcription
        
        groq_data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 700
        }
        groq_response = requests.post(GROQ_URL, headers=groq_headers, json=groq_data, timeout=90)
        groq_response.raise_for_status()

        groq_response_json = groq_response.json()
        answer = groq_response_json["choices"][0]["message"]["content"]
        print(f"[DEBUG] Groq Response (Answer): {answer[:100]}...")

        # ElevenLabs ile ses oluştur
        elevenlabs_url = f"{ELEVENLABS_BASE_URL}/{ELEVENLABS_VOICE_ID}"
        elevenlabs_headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        elevenlabs_data = {
            "text": answer,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": DEFAULT_VOICE_SETTINGS
        }
        elevenlabs_response = requests.post(elevenlabs_url, headers=elevenlabs_headers, json=elevenlabs_data, timeout=120)
        elevenlabs_response.raise_for_status()

        audio_data = elevenlabs_response.content
        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ElevenLabs API'den boş ses verisi alındı."
            )

        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join("static", audio_filename)
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        audio_url = f"/static/{audio_filename}"

        # Konuşma geçmişine kaydet
        conversation_manager.save_conversation(transcription, answer, current_personality, audio_url)

        return JSONResponse({"answer": answer, "audio_url": audio_url})

    except requests.exceptions.RequestException as e:
        error_detail = e.response.text if e.response is not None else str(e)
        print(f"[ERROR] API Request Error: {error_detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"API bağlantı/istek hatası: {error_detail}"
        )
    except Exception as e:
        print(f"[ERROR] Unexpected Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Beklenmedik hata: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("[ERROR] index.html not found!")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="index.html dosyası bulunamadı.")
    except Exception as e:
        print(f"[ERROR] Error reading index.html: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"index.html okunurken hata: {e}")
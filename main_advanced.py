import os
from dotenv import load_dotenv
import uuid
from fastapi import FastAPI, HTTPException, status, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
from groq import Groq
import json
from datetime import datetime, timedelta
import sqlite3
from typing import List, Optional, Dict, Any
import asyncio
import time
from functools import wraps
import numpy as np
from textblob import TextBlob
import librosa
import io
import base64
from collections import defaultdict
import threading
import queue

app = FastAPI()

# Statik dosyaları sunmak için
app.mount("/static", StaticFiles(directory="static"), name="static")

# .env dosyasını yükle
load_dotenv()

# Ortam değişkenleri
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# API URL'leri
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# Groq istemcisini başlat
client = Groq(api_key=GROQ_API_KEY)

# WebSocket bağlantı yöneticisi
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_sessions: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_sessions[user_id] = websocket
    
    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Kullanıcı session'ını temizle
        for user_id, ws in self.user_sessions.items():
            if ws == websocket:
                del self.user_sessions[user_id]
                break
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                await self.disconnect(connection)
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_sessions:
            try:
                await self.user_sessions[user_id].send_text(message)
            except:
                await self.disconnect(self.user_sessions[user_id])

manager = ConnectionManager()

# Çoklu AI model desteği
AI_MODELS = {
    "groq_llama": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "url": GROQ_URL,
        "api_key": GROQ_API_KEY
    },
    "groq_mixtral": {
        "provider": "groq", 
        "model": "mixtral-8x7b-32768",
        "url": GROQ_URL,
        "api_key": GROQ_API_KEY
    },
    "openai_gpt4": {
        "provider": "openai",
        "model": "gpt-4",
        "url": OPENAI_URL,
        "api_key": OPENAI_API_KEY
    },
    "anthropic_claude": {
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "url": ANTHROPIC_URL,
        "api_key": ANTHROPIC_API_KEY
    }
}

# Akıllı context yönetimi
class ConversationContext:
    def __init__(self, max_tokens=4000):
        self.context_window = []
        self.max_tokens = max_tokens
        self.memory = []
        self.emotion_history = []
    
    def add_to_context(self, user_input: str, ai_response: str, emotion: Dict = None):
        self.context_window.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(),
            "emotion": emotion
        })
        self.context_window.append({
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.now()
        })
        
        # Context window'u optimize et
        if len(self.context_window) > 10:
            self.context_window = self.context_window[-10:]
    
    def get_context_for_model(self, model_name: str) -> List[Dict]:
        # Model'e göre context formatını ayarla
        if "anthropic" in model_name:
            return self.format_for_anthropic()
        else:
            return self.format_for_openai()
    
    def format_for_openai(self) -> List[Dict]:
        return [{"role": msg["role"], "content": msg["content"]} 
                for msg in self.context_window]
    
    def format_for_anthropic(self) -> List[Dict]:
        # Anthropic formatı için dönüştür
        messages = []
        for msg in self.context_window:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        return messages

# Duygu analizi servisi
class EmotionAnalyzer:
    def __init__(self):
        self.emotion_keywords = {
            "mutlu": ["güzel", "harika", "mükemmel", "sevindim", "mutlu"],
            "üzgün": ["kötü", "üzgün", "kederli", "mutsuz", "kırgın"],
            "kızgın": ["sinirli", "kızgın", "öfkeli", "rahatsız", "bıktım"],
            "endişeli": ["endişe", "kaygı", "stres", "panik", "korku"],
            "sakin": ["sakin", "rahat", "huzurlu", "dingin", "sessiz"]
        }
    
    def analyze_text_emotion(self, text: str) -> Dict:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Türkçe duygu analizi
        emotion_score = self.analyze_turkish_emotion(text)
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "emotion": self.classify_emotion(polarity, emotion_score),
            "confidence": abs(polarity) + abs(subjectivity)
        }
    
    def analyze_turkish_emotion(self, text: str) -> Dict:
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        return emotion_scores
    
    def classify_emotion(self, polarity: float, emotion_scores: Dict) -> str:
        if polarity > 0.3:
            return "mutlu"
        elif polarity < -0.3:
            return "üzgün"
        elif emotion_scores.get("kızgın", 0) > 0:
            return "kızgın"
        elif emotion_scores.get("endişeli", 0) > 0:
            return "endişeli"
        else:
            return "nötr"

# Performans izleme
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def log_metric(self, function_name: str, execution_time: float, success: bool, 
                   additional_data: Dict = None):
        metric = {
            "function": function_name,
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.now(),
            "additional_data": additional_data or {}
        }
        self.metrics[function_name].append(metric)
    
    def get_performance_stats(self) -> Dict:
        stats = {}
        for func_name, metrics in self.metrics.items():
            if metrics:
                execution_times = [m["execution_time"] for m in metrics]
                success_rate = sum(1 for m in metrics if m["success"]) / len(metrics)
                
                stats[func_name] = {
                    "avg_execution_time": np.mean(execution_times),
                    "max_execution_time": np.max(execution_times),
                    "min_execution_time": np.min(execution_times),
                    "success_rate": success_rate,
                    "total_calls": len(metrics)
                }
        return stats

# Kullanım analitikleri
class AnalyticsService:
    def __init__(self):
        self.db_path = 'analytics.db'
        self.init_analytics_db()
    
    def init_analytics_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                event_type TEXT,
                event_data TEXT,
                timestamp TEXT,
                session_id TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    async def track_event(self, user_id: str, event_type: str, event_data: Dict, session_id: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analytics (user_id, event_type, event_data, timestamp, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, event_type, json.dumps(event_data), datetime.now().isoformat(), session_id))
        conn.commit()
        conn.close()
    
    async def get_analytics(self, days: int = 30) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Son N günün verilerini al
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Toplam kullanım
        cursor.execute('''
            SELECT COUNT(*) FROM analytics WHERE timestamp >= ?
        ''', (since_date,))
        total_events = cursor.fetchone()[0]
        
        # Event türlerine göre dağılım
        cursor.execute('''
            SELECT event_type, COUNT(*) FROM analytics 
            WHERE timestamp >= ? GROUP BY event_type
        ''', (since_date,))
        event_distribution = dict(cursor.fetchall())
        
        # Aktif kullanıcılar
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) FROM analytics WHERE timestamp >= ?
        ''', (since_date,))
        active_users = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_events": total_events,
            "event_distribution": event_distribution,
            "active_users": active_users,
            "period_days": days
        }

# Gerçek zamanlı ses işleme
class RealTimeAudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
    
    def start_processing(self):
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_audio_loop)
        self.processing_thread.start()
    
    def stop_processing(self):
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def _process_audio_loop(self):
        while self.is_processing:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                self._process_audio_chunk(audio_data)
            except queue.Empty:
                continue
    
    def _process_audio_chunk(self, audio_data: bytes):
        # Ses işleme algoritmaları
        # 1. Gürültü filtreleme
        filtered_audio = self._apply_noise_filter(audio_data)
        
        # 2. Echo cancellation
        processed_audio = self._apply_echo_cancellation(filtered_audio)
        
        # 3. Ses seviyesi normalizasyonu
        normalized_audio = self._normalize_audio(processed_audio)
        
        return normalized_audio
    
    def _apply_noise_filter(self, audio_data: bytes) -> bytes:
        # Basit gürültü filtreleme
        # Gerçek uygulamada daha gelişmiş algoritmalar kullanılabilir
        return audio_data
    
    def _apply_echo_cancellation(self, audio_data: bytes) -> bytes:
        # Echo cancellation
        return audio_data
    
    def _normalize_audio(self, audio_data: bytes) -> bytes:
        # Ses seviyesi normalizasyonu
        return audio_data

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
            session_id TEXT,
            emotion TEXT,
            model_used TEXT,
            response_time REAL,
            user_satisfaction INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# Veritabanını başlat
init_database()

# Global servisler
performance_monitor = PerformanceMonitor()
analytics_service = AnalyticsService()
emotion_analyzer = EmotionAnalyzer()
audio_processor = RealTimeAudioProcessor()

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
    },
    "teacher": {
        "system_prompt": """Sen deneyimli bir öğretmensin. Karmaşık konuları basit ve anlaşılır şekilde açıklarsın. 
        Örnekler verir, pratik uygulamalar önerir ve öğrenmeyi kolaylaştırırsın. 
        Türkçe konuş ve pedagojik yaklaşımlar kullan."""
    },
    "doctor": {
        "system_prompt": """Sen uzman bir doktorsun. Sağlık konularında bilimsel ve güvenilir bilgiler verirsin. 
        Semptomları analiz eder, önleyici tedbirler önerir ve sağlıklı yaşam tarzları önerirsin. 
        Türkçe konuş ve tıbbi terimleri açıkla."""
    }
}

# Varsayılan ayarlar
current_personality = "engineer"
current_model = "groq_llama"

# Ses kalitesi ayarları
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True
}

# Pydantic modelleri
class QuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class PersonalityRequest(BaseModel):
    personality: str

class ModelRequest(BaseModel):
    model: str

class VoiceSettingsRequest(BaseModel):
    stability: Optional[float] = 0.5
    similarity_boost: Optional[float] = 0.75
    style: Optional[float] = 0.0
    use_speaker_boost: Optional[bool] = True

class SessionRequest(BaseModel):
    session_id: str

# Performans izleme decorator'ı
def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            performance_monitor.log_metric(
                func.__name__, 
                execution_time, 
                True,
                {"args": str(args), "kwargs": str(kwargs)}
            )
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            performance_monitor.log_metric(
                func.__name__, 
                execution_time, 
                False,
                {"error": str(e), "args": str(args), "kwargs": str(kwargs)}
            )
            raise
    return wrapper

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

# AI model seçimi ve çağrısı
async def call_ai_model(model_name: str, messages: List[Dict], system_prompt: str) -> str:
    model_config = AI_MODELS.get(model_name, AI_MODELS["groq_llama"])
    
    if model_config["provider"] == "groq":
        return await call_groq_model(model_config, messages, system_prompt)
    elif model_config["provider"] == "openai":
        return await call_openai_model(model_config, messages, system_prompt)
    elif model_config["provider"] == "anthropic":
        return await call_anthropic_model(model_config, messages, system_prompt)
    else:
        raise ValueError(f"Unsupported model provider: {model_config['provider']}")

async def call_groq_model(model_config: Dict, messages: List[Dict], system_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {model_config['api_key']}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_config["model"],
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "max_tokens": 700
    }
    
    response = requests.post(model_config["url"], headers=headers, json=data, timeout=90)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

async def call_openai_model(model_config: Dict, messages: List[Dict], system_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {model_config['api_key']}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_config["model"],
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "max_tokens": 700
    }
    
    response = requests.post(model_config["url"], headers=headers, json=data, timeout=90)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

async def call_anthropic_model(model_config: Dict, messages: List[Dict], system_prompt: str) -> str:
    headers = {
        "x-api-key": model_config['api_key'],
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    # Anthropic formatına dönüştür
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            formatted_messages.append({"role": "assistant", "content": msg["content"]})
    
    data = {
        "model": model_config["model"],
        "messages": formatted_messages,
        "system": system_prompt,
        "max_tokens": 700
    }
    
    response = requests.post(model_config["url"], headers=headers, json=data, timeout=90)
    response.raise_for_status()
    
    result = response.json()
    return result["content"][0]["text"]

# WebSocket endpoint'i
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Gerçek zamanlı işlem
            if message["type"] == "voice_input":
                # Ses verisi işleme
                audio_data = base64.b64decode(message["audio"])
                processed_audio = audio_processor._process_audio_chunk(audio_data)
                
                # AI yanıtı al
                response = await process_realtime_request(message["text"], user_id)
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "ai_response",
                        "text": response["answer"],
                        "audio_url": response["audio_url"],
                        "emotion": response["emotion"]
                    }), 
                    user_id
                )
            
            elif message["type"] == "voice_command":
                # Sesli komut işleme
                command = message["command"]
                response = await process_voice_command(command, user_id)
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "command_response",
                        "response": response
                    }), 
                    user_id
                )
    
    except WebSocketDisconnect:
        await manager.disconnect(websocket)

# Sesli komut işleme
async def process_voice_command(command: str, user_id: str) -> Dict:
    command_lower = command.lower()
    
    if "durdur" in command_lower:
        return {"action": "stop_recording", "message": "Kayıt durduruldu"}
    elif "tekrarla" in command_lower:
        return {"action": "repeat_last", "message": "Son yanıt tekrarlanıyor"}
    elif "daha yavaş" in command_lower:
        return {"action": "slow_speech", "message": "Konuşma hızı yavaşlatıldı"}
    elif "daha hızlı" in command_lower:
        return {"action": "fast_speech", "message": "Konuşma hızı artırıldı"}
    elif "geçmiş" in command_lower:
        return {"action": "show_history", "message": "Geçmiş gösteriliyor"}
    elif "ayarlar" in command_lower:
        return {"action": "show_settings", "message": "Ayarlar açılıyor"}
    else:
        return {"action": "unknown", "message": "Komut anlaşılamadı"}

# Gerçek zamanlı istek işleme
async def process_realtime_request(text: str, user_id: str) -> Dict:
    # Duygu analizi
    emotion = emotion_analyzer.analyze_text_emotion(text)
    
    # AI yanıtı al
    system_prompt = PERSONALITIES[current_personality]["system_prompt"]
    messages = [{"role": "user", "content": text}]
    
    answer = await call_ai_model(current_model, messages, system_prompt)
    
    # Ses oluştur
    audio_url = await generate_speech(answer)
    
    # Analitik kaydet
    await analytics_service.track_event(
        user_id, 
        "realtime_conversation", 
        {
            "text": text,
            "answer": answer,
            "emotion": emotion,
            "personality": current_personality,
            "model": current_model
        }
    )
    
    return {
        "answer": answer,
        "audio_url": audio_url,
        "emotion": emotion
    }

# Ses oluşturma
async def generate_speech(text: str) -> str:
    try:
        elevenlabs_url = f"{ELEVENLABS_BASE_URL}/{ELEVENLABS_VOICE_ID}"
        elevenlabs_headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        elevenlabs_data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": DEFAULT_VOICE_SETTINGS
        }
        
        elevenlabs_response = requests.post(
            elevenlabs_url, 
            headers=elevenlabs_headers, 
            json=elevenlabs_data, 
            timeout=120
        )
        elevenlabs_response.raise_for_status()
        
        audio_data = elevenlabs_response.content
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join("static", audio_filename)
        
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        return f"/static/{audio_filename}"
    
    except Exception as e:
        print(f"Speech generation error: {e}")
        return None

# API Endpoint'leri
@app.post("/set-personality")
@monitor_performance
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

@app.post("/set-model")
@monitor_performance
async def set_model(request: ModelRequest):
    global current_model
    model = request.model
    
    if model not in AI_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geçersiz model: {model}. Geçerli seçenekler: {list(AI_MODELS.keys())}"
        )
    
    current_model = model
    return JSONResponse({"message": f"Model {model} olarak ayarlandı"})

@app.post("/set-voice-settings")
@monitor_performance
async def set_voice_settings(request: VoiceSettingsRequest):
    global DEFAULT_VOICE_SETTINGS
    DEFAULT_VOICE_SETTINGS.update(request.dict(exclude_unset=True))
    return JSONResponse({"message": "Ses ayarları güncellendi", "settings": DEFAULT_VOICE_SETTINGS})

@app.get("/performance-stats")
async def get_performance_stats():
    return JSONResponse(performance_monitor.get_performance_stats())

@app.get("/analytics")
async def get_analytics(days: int = 30):
    analytics = await analytics_service.get_analytics(days)
    return JSONResponse(analytics)

@app.get("/models")
async def get_available_models():
    return JSONResponse({
        "models": list(AI_MODELS.keys()),
        "current_model": current_model
    })

@app.get("/personalities")
async def get_available_personalities():
    return JSONResponse({
        "personalities": list(PERSONALITIES.keys()),
        "current_personality": current_personality
    })

# Ana endpoint'ler
@app.post("/ask")
@monitor_performance
async def ask(request_data: QuestionRequest):
    start_time = time.time()
    
    # Duygu analizi
    emotion = emotion_analyzer.analyze_text_emotion(request_data.question)
    
    # Web search
    web_results = None
    if any(keyword in request_data.question.lower() for keyword in ["güncel", "son", "haber", "bugün", "recent", "latest", "news"]):
        web_results = await web_search(request_data.question)
    
    # AI yanıtı al
    system_prompt = PERSONALITIES[current_personality]["system_prompt"]
    
    if web_results:
        context = "Güncel web arama sonuçları:\n"
        for i, result in enumerate(web_results, 1):
            context += f"{i}. {result['title']}\n{result['description']}\nURL: {result['url']}\n\n"
        
        user_content = f"Kullanıcı sorusu: {request_data.question}\n\n{context}\nBu bilgileri kullanarak cevap ver."
    else:
        user_content = request_data.question
    
    messages = [{"role": "user", "content": user_content}]
    answer = await call_ai_model(current_model, messages, system_prompt)
    
    # Ses oluştur
    audio_url = await generate_speech(answer)
    
    response_time = time.time() - start_time
    
    # Analitik kaydet
    if request_data.user_id:
        await analytics_service.track_event(
            request_data.user_id,
            "conversation",
            {
                "question": request_data.question,
                "answer": answer,
                "personality": current_personality,
                "model": current_model,
                "emotion": emotion,
                "response_time": response_time,
                "web_search_used": web_results is not None
            },
            request_data.session_id
        )
    
    return JSONResponse({
        "answer": answer,
        "audio_url": audio_url,
        "emotion": emotion,
        "response_time": response_time,
        "model_used": current_model
    })

@app.post("/upload-audio")
@monitor_performance
async def upload_audio(file: UploadFile = File(...), user_id: str = None, session_id: str = None):
    start_time = time.time()
    
    try:
        # Ses dosyasını oku ve işle
        audio_content = await file.read()
        processed_audio = audio_processor._process_audio_chunk(audio_content)
        
        # Ses dosyasını kaydet
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join("static", audio_filename)
        with open(audio_path, "wb") as f:
            f.write(processed_audio)
        
        # Groq Whisper ile sesi metne çevir
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        
        # Duygu analizi
        emotion = emotion_analyzer.analyze_text_emotion(transcription)
        
        # Web search
        web_results = None
        if any(keyword in transcription.lower() for keyword in ["güncel", "son", "haber", "bugün", "recent", "latest", "news"]):
            web_results = await web_search(transcription)
        
        # AI yanıtı al
        system_prompt = PERSONALITIES[current_personality]["system_prompt"]
        
        if web_results:
            context = "Güncel web arama sonuçları:\n"
            for i, result in enumerate(web_results, 1):
                context += f"{i}. {result['title']}\n{result['description']}\nURL: {result['url']}\n\n"
            
            user_content = f"Kullanıcı sorusu: {transcription}\n\n{context}\nBu bilgileri kullanarak cevap ver."
        else:
            user_content = transcription
        
        messages = [{"role": "user", "content": user_content}]
        answer = await call_ai_model(current_model, messages, system_prompt)
        
        # Ses oluştur
        audio_url = await generate_speech(answer)
        
        response_time = time.time() - start_time
        
        # Analitik kaydet
        if user_id:
            await analytics_service.track_event(
                user_id,
                "voice_conversation",
                {
                    "transcription": transcription,
                    "answer": answer,
                    "personality": current_personality,
                    "model": current_model,
                    "emotion": emotion,
                    "response_time": response_time,
                    "web_search_used": web_results is not None
                },
                session_id
            )
        
        return JSONResponse({
            "answer": answer,
            "audio_url": audio_url,
            "emotion": emotion,
            "response_time": response_time,
            "model_used": current_model
        })
    
    except Exception as e:
        print(f"Audio processing error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
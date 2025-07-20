# 🚀 AI Destekli Sesli Soru-Cevap Uygulaması - Gelişmiş Versiyon

Bu proje, en son AI teknolojilerini kullanarak geliştirilmiş kapsamlı bir sesli soru-cevap uygulamasıdır. Çoklu AI model desteği, gerçek zamanlı ses işleme, duygu analizi ve gelişmiş analitik özellikleri içerir.

## 🌟 **Yeni Özellikler**

### 🤖 **Çoklu AI Model Desteği**
- **Groq Llama**: Hızlı ve etkili yanıtlar
- **Groq Mixtral**: Gelişmiş anlayış ve analiz
- **OpenAI GPT-4**: En gelişmiş AI modeli
- **Anthropic Claude**: Güvenli ve güvenilir yanıtlar

### 🎵 **Gelişmiş Ses İşleme**
- **Gerçek Zamanlı Gürültü Filtreleme**: Echo cancellation, noise suppression
- **Ses Kalitesi Optimizasyonu**: Otomatik gain control
- **Gelişmiş Ses Ayarları**: Stabilite, benzerlik, stil kontrolü
- **Çoklu Ses Formatı Desteği**: WebM, MP3, WAV

### 😊 **Duygu Analizi ve Ses Tonu**
- **Metin Duygu Analizi**: TextBlob ile gelişmiş analiz
- **Ses Duygu Algılama**: Librosa ile ses analizi
- **Duygu Geçmişi**: Kullanıcı duygu durumu takibi
- **Duygu Bazlı Yanıtlar**: Duyguya uygun AI yanıtları

### 📊 **Gelişmiş Analitik ve İzleme**
- **Performans İzleme**: Her işlem için detaylı metrikler
- **Kullanım Analitikleri**: Kullanıcı davranış analizi
- **Gerçek Zamanlı İstatistikler**: Canlı kullanım verileri
- **Model Performans Karşılaştırması**: AI modellerinin karşılaştırması

### 🔄 **WebSocket Gerçek Zamanlı İletişim**
- **Anlık Ses İşleme**: Gerçek zamanlı ses analizi
- **Canlı Yanıtlar**: Anında AI yanıtları
- **Bağlantı Durumu**: WebSocket bağlantı izleme
- **Otomatik Yeniden Bağlanma**: Bağlantı kopması durumunda

### 🎨 **Gelişmiş Arayüz Animasyonları**
- **Smooth Transitions**: Yumuşak geçiş animasyonları
- **Ses Dalgası Görselleştirme**: Gerçek zamanlı ses dalgası
- **Duygu Göstergeleri**: Renkli duygu durumu göstergeleri
- **Responsive Tasarım**: Mobil uyumlu arayüz

### 🎤 **Sesli Komutlar ve Kontroller**
- **Sesli Komutlar**: "Durdur", "Tekrarla", "Daha yavaş"
- **Otomatik Komut Algılama**: Sesli komut tanıma
- **Hız Kontrolü**: Ses hızı ayarlama
- **Geçmiş Kontrolü**: Sesli geçmiş yönetimi

### 🧠 **Akıllı Context Yönetimi**
- **Dinamik Context**: Otomatik context optimizasyonu
- **Model Bazlı Format**: Her AI modeli için özel format
- **Context Boyutu Yönetimi**: Otomatik boyut kontrolü
- **Geçmiş Hafıza**: Önceki konuşmaları hatırlama

## 🛠️ **Teknolojiler**

### **Backend**
- **FastAPI**: Modern Python web framework
- **WebSocket**: Gerçek zamanlı iletişim
- **SQLite**: Hafif veritabanı
- **Redis**: Cache ve session yönetimi

### **AI ve ML**
- **Groq API**: Hızlı AI modelleri
- **OpenAI API**: GPT-4 entegrasyonu
- **Anthropic API**: Claude entegrasyonu
- **TextBlob**: Duygu analizi
- **Librosa**: Ses analizi

### **Ses İşleme**
- **ElevenLabs**: Gelişmiş ses sentezi
- **Web Audio API**: Tarayıcı ses işleme
- **Pydub**: Ses dosyası işleme
- **Scipy**: Bilimsel hesaplamalar

### **Frontend**
- **HTML5/CSS3**: Modern arayüz
- **JavaScript ES6+**: Gelişmiş etkileşim
- **Canvas API**: Ses dalgası çizimi
- **WebSocket API**: Gerçek zamanlı iletişim

## 📋 **Kurulum**

### 1. **Gereksinimler**
```bash
# Python 3.8+ gerekli
python --version

# Gelişmiş bağımlılıkları yükle
pip install -r requirements_advanced.txt
```

### 2. **API Anahtarları**
```env
# .env dosyasına ekleyin
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_voice_id
BRAVE_API_KEY=your_brave_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 3. **Uygulamayı Çalıştırma**
```bash
# Gelişmiş versiyonu çalıştır
python main_advanced.py

# Veya uvicorn ile
uvicorn main_advanced:app --reload --host 0.0.0.0 --port 8000
```

## 🎯 **Kullanım**

### **Temel Kullanım**
1. Tarayıcıda `http://localhost:8000` adresine gidin
2. İstediğiniz AI modelini seçin
3. Personality butonlarından birini seçin
4. "Başlat" butonuna basın ve konuşun
5. AI'ın sesli yanıtını dinleyin

### **Gelişmiş Özellikler**

#### **AI Model Seçimi**
- **Groq Llama**: Hızlı yanıtlar için
- **Groq Mixtral**: Detaylı analiz için
- **OpenAI GPT-4**: En gelişmiş yanıtlar için
- **Anthropic Claude**: Güvenli yanıtlar için

#### **Ses Ayarları**
- **Stabilite**: Ses tutarlılığı (0.0-1.0)
- **Benzerlik Artırma**: Ses benzerliği (0.0-1.0)
- **Stil**: Ses karakteri (0.0-1.0)
- **Konuşmacı Artırma**: Ses kalitesi optimizasyonu

#### **Duygu Analizi**
- **Mutlu**: Pozitif duygular
- **Üzgün**: Negatif duygular
- **Kızgın**: Öfke durumu
- **Endişeli**: Kaygı durumu
- **Nötr**: Tarafsız durum

#### **Sesli Komutlar**
- **"Durdur"**: Kayıt durdurma
- **"Tekrarla"**: Son yanıtı tekrarlama
- **"Daha yavaş"**: Konuşma hızını yavaşlatma
- **"Daha hızlı"**: Konuşma hızını artırma
- **"Geçmiş"**: Geçmiş panelini açma
- **"Ayarlar"**: Ayarlar panelini açma

## 📊 **API Endpoint'leri**

### **Temel Endpoint'ler**
```http
POST /ask                    # Metin tabanlı soru
POST /upload-audio          # Ses tabanlı soru
POST /set-personality       # Personality değiştirme
POST /set-model            # AI model değiştirme
POST /set-voice-settings   # Ses ayarları
```

### **Analitik Endpoint'ler**
```http
GET /performance-stats      # Performans istatistikleri
GET /analytics             # Kullanım analitikleri
GET /models                # Mevcut AI modelleri
GET /personalities         # Mevcut personality'ler
```

### **WebSocket Endpoint'leri**
```http
WS /ws/{user_id}          # Gerçek zamanlı iletişim
```

### **Geçmiş Yönetimi**
```http
GET /conversation-history  # Konuşma geçmişi
DELETE /conversation/{id}  # Tek konuşma silme
DELETE /clear-history      # Tüm geçmişi temizleme
```

## 🔧 **Geliştirme**

### **Yeni AI Model Ekleme**
```python
AI_MODELS = {
    "yeni_model": {
        "provider": "yeni_provider",
        "model": "model_adi",
        "url": "api_url",
        "api_key": "api_key"
    }
}
```

### **Yeni Personality Ekleme**
```python
PERSONALITIES = {
    "yeni_personality": {
        "system_prompt": "Yeni personality'nin sistem prompt'u"
    }
}
```

### **Yeni Duygu Kategorisi**
```python
emotion_keywords = {
    "yeni_duygu": ["anahtar", "kelimeler", "listesi"]
}
```

## 📈 **Performans Optimizasyonları**

### **Ses İşleme**
- **Echo Cancellation**: Yansıma giderme
- **Noise Suppression**: Gürültü filtreleme
- **Auto Gain Control**: Otomatik ses seviyesi
- **Real-time Processing**: Gerçek zamanlı işleme

### **AI Model Seçimi**
- **Hız Optimizasyonu**: Groq Llama
- **Kalite Optimizasyonu**: GPT-4
- **Güvenlik**: Claude
- **Maliyet**: Mixtral

### **Cache Sistemi**
- **Redis Cache**: Hızlı yanıtlar
- **Response Caching**: Tekrarlanan sorular
- **Model Caching**: AI model cache'i

## 🚀 **Deployment**

### **Docker ile Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_advanced.txt .
RUN pip install -r requirements_advanced.txt
COPY . .
EXPOSE 8000
CMD ["python", "main_advanced.py"]
```

### **Environment Variables**
```bash
export GROQ_API_KEY="your_key"
export ELEVENLABS_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
```

## 📊 **Monitoring ve Analytics**

### **Performans Metrikleri**
- **Response Time**: Yanıt süreleri
- **Success Rate**: Başarı oranları
- **Error Rate**: Hata oranları
- **Model Usage**: Model kullanım istatistikleri

### **Kullanım Analitikleri**
- **User Sessions**: Kullanıcı oturumları
- **Popular Personalities**: Popüler personality'ler
- **Model Performance**: Model performansları
- **Voice Quality**: Ses kalitesi metrikleri

## 🔒 **Güvenlik**

### **API Güvenliği**
- **Rate Limiting**: İstek sınırlama
- **Input Validation**: Girdi doğrulama
- **Error Handling**: Hata yönetimi
- **Logging**: Güvenlik logları

### **Veri Güvenliği**
- **Encryption**: Veri şifreleme
- **Secure Storage**: Güvenli saklama
- **Access Control**: Erişim kontrolü
- **Audit Trail**: Denetim izi

## 🤝 **Katkıda Bulunma**

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📝 **Lisans**

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 **İletişim**

- **GitHub**: [Repository Linki]
- **Email**: [Email Adresi]
- **Issues**: [GitHub Issues]

## 🎉 **Teşekkürler**

Bu proje aşağıdaki teknolojiler ve servisler sayesinde mümkün olmuştur:

- **Groq**: Hızlı AI modelleri
- **OpenAI**: GPT-4 API
- **Anthropic**: Claude API
- **ElevenLabs**: Ses sentezi
- **FastAPI**: Web framework
- **WebSocket**: Gerçek zamanlı iletişim

---

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!** 
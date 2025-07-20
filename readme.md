# AI Destekli Sesli Soru-Cevap Uygulaması

Bu uygulama, kullanıcıların sesli olarak soru sorabilmesini ve AI'ın sesli yanıt vermesini sağlayan modern bir web uygulamasıdır. Farklı kişilik modları ve güncel web arama özelliği ile zenginleştirilmiştir.

## 🚀 Özellikler

- **Sesli Sohbet**: Mikrofon ile konuşarak soru sorabilir, AI'dan sesli yanıt alabilirsiniz
- **Otomatik Diyalog**: Tek butona basarak sürekli konuşma modunu başlatabilirsiniz
- **Farklı Kişilikler**: Engineer, Scientist ve Planner modları ile farklı yaklaşımlar
- **Güncel Bilgi**: Brave Search API ile gerçek zamanlı web araması
- **Çok Dilli Destek**: Türkçe ve İngilizce dil desteği
- **Modern Arayüz**: Kullanıcı dostu ve responsive tasarım

## 🛠️ Teknolojiler

- **Backend**: FastAPI (Python)
- **AI Model**: Groq Llama-3.1-8b-instant
- **Ses Tanıma**: Groq Whisper
- **Ses Sentezi**: ElevenLabs
- **Web Arama**: Brave Search API
- **Frontend**: HTML5, CSS3, JavaScript

## 📋 Gereksinimler

- Python 3.8+
- Mikrofon erişimi
- API anahtarları (aşağıda açıklanmıştır)

## ⚙️ Kurulum

### 1. Projeyi İndirin
```bash
git clone <repository-url>
cd soru_cevap_uygulamasi
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install fastapi uvicorn python-dotenv requests groq
```

### 3. API Anahtarlarını Ayarlayın

`env.example` dosyasını `.env` olarak kopyalayın ve API anahtarlarınızı ekleyin:

```bash
cp env.example .env
```

`.env` dosyasını düzenleyin:
```env
GROQ_API_KEY=your_groq_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id_here
BRAVE_API_KEY=your_brave_api_key_here
```

### 4. API Anahtarlarını Alın

#### Groq API Key
1. [Groq Console](https://console.groq.com/keys) adresine gidin
2. Hesap oluşturun veya giriş yapın
3. API key oluşturun

#### ElevenLabs API Key
1. [ElevenLabs](https://elevenlabs.io/account) adresine gidin
2. Hesap oluşturun veya giriş yapın
3. API key alın
4. [Voice Library](https://elevenlabs.io/voice-library) adresinden bir ses seçin ve Voice ID'sini kopyalayın

#### Brave Search API Key
1. [Brave Search API](https://api.search.brave.com/) adresine gidin
2. API key alın

### 5. Uygulamayı Çalıştırın
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Uygulama http://localhost:8000 adresinde çalışacaktır.

## 🎯 Kullanım

### Temel Kullanım
1. Tarayıcınızda http://localhost:8000 adresine gidin
2. İstediğiniz kişilik modunu seçin (Engineer, Scientist, Planner)
3. "Başlat" butonuna basın
4. Mikrofon izni verin
5. Sorunuzu sorun
6. AI'ın sesli yanıtını dinleyin

### Kişilik Modları

#### Engineer
- Teknik konularda detaylı çözümler
- Kod örnekleri ve performans optimizasyonları
- En iyi pratikler ve teknik terimler

#### Scientist
- Bilimsel analiz ve araştırma yöntemleri
- Kanıta dayalı yaklaşımlar
- Karmaşık kavramları basit açıklamalar

#### Planner
- Stratejik planlama ve organizasyon
- Adım adım analiz
- Proje ve zaman yönetimi

### Güncel Bilgi Arama
"güncel", "son", "haber", "bugün" gibi kelimeler içeren sorularda otomatik olarak web araması yapılır.

## 🔧 Geliştirme

### Proje Yapısı
```
soru_cevap_uygulamasi/
├── main.py              # FastAPI backend
├── index.html           # Frontend arayüzü
├── static/              # Ses dosyaları
├── .env                 # API anahtarları (git'e eklenmez)
├── env.example          # API anahtarları örneği
└── README.md           # Bu dosya
```

### Yeni Kişilik Ekleme
`main.py` dosyasındaki `PERSONALITIES` sözlüğüne yeni kişilik ekleyebilirsiniz:

```python
PERSONALITIES = {
    "yeni_kisilik": {
        "system_prompt": "Yeni kişiliğin sistem prompt'u buraya"
    }
}
```

## 🐛 Sorun Giderme

### Mikrofon Erişimi
- Tarayıcı ayarlarından mikrofon iznini kontrol edin
- HTTPS kullanıyorsanız sertifika sorunları olabilir

### API Hataları
- API anahtarlarının doğru olduğundan emin olun
- API limitlerini kontrol edin
- Ağ bağlantınızı kontrol edin

### Ses Sorunları
- Tarayıcı ses ayarlarını kontrol edin
- ElevenLabs Voice ID'sinin doğru olduğundan emin olun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📞 İletişim

Sorularınız için issue açabilir veya iletişime geçebilirsiniz.

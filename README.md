# 🛍️ AI Retail Assistant (Akıllı Alışveriş Asistanı)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![Gemini](https://img.shields.io/badge/Google%20Gemini-Pro-blueviolet.svg)

Bu proje, Makine Öğrenmesi (Derin Öğrenme) ve Üretken Yapay Zeka'nın (LLM) gücünü birleştiren yenilikçi bir **Hibrit Yapay Zeka Alışveriş Asistanıdır**. Müşterilerin doğal dilde yazdıkları mesajları analiz ederek, yaşlarına, konumlarına (AVM) ve alışveriş tarihlerine göre en uygun ürün kategorisini ve tahmini harcama bütçesini önerir.

Geleneksel chatbot'ların aksine kural tabanlı değil, **veri odaklı çalışır.** Keras ile eğitilmiş MLP (Çok Katmanlı Algılayıcı) modelleri tahminlemeyi yaparken, Google Gemini verileri işleme ve doğal dilde cevap üretme görevini üstlenir.

## ✨ Öne Çıkan Özellikler

* **Varlık Çıkarımı (Entity Extraction):** Google Gemini kullanılarak, kullanıcının serbest metninden yaş, cinsiyet, alışveriş merkezi ve gün tipi gibi değişkenler JSON formatında otomatik olarak ayrıştırılır.
* **Derin Öğrenme ile Sınıflandırma:** Keras ile oluşturulan MLP Sınıflandırma modeli, çıkarılan varlıkları kullanarak müşterinin hangi ürün kategorisine (Giyim, Teknoloji, Kozmetik vb.) ilgi duyacağını tahmin eder.
* **Derin Öğrenme ile Fiyat Tahmini:** Ayrı bir MLP Regresyon modeli, müşterinin ilgili kategoride ortalama ne kadar harcama yapacağını TL cinsinden hesaplar.
* **Müşteri Segmentasyonu:** K-Means kümeleme algoritması kullanılarak veritabanındaki müşteriler davranışlarına göre profillenmiştir.
* **Modern Web Arayüzü:** FastAPI ile oluşturulan backend, WhatsApp benzeri şık, mobil uyumlu ve asenkron bir web arayüzü ile sunulur.

## 🧠 Sistem Mimarisi

Sistem şu sırayla çalışır:
1. **Kullanıcı Girdisi:** *"25 yaşındayım, bugün Kanyon'dayım bana bir şeyler öner."*
2. **LLM Parsing:** Gemini bu cümleyi `{age: 25, mall: "Kanyon AVM", intent: "recommendation"}` şeklinde ayrıştırır.
3. **Vektörizasyon:** Çıkarılan bu JSON verisi, Label Encoder ve StandardScaler ile modelin anlayacağı sayısal tensörlere dönüştürülür.
4. **AI Tahmini:** `.h5` formatında kaydedilmiş Keras modelleri, kategoriyi ve bütçeyi tahmin eder.
5. **Generative Response:** Model çıktıları tekrar Gemini'ye beslenir ve müşteriye özel, doğal ve akıcı bir son cevap üretilir.

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
Projeyi yerel makinenizde veya Google Colab üzerinde çalıştırabilirsiniz.
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn fastapi uvicorn pyngrok nest_asyncio google-generativeai

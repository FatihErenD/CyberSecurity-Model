# BÜYÜK DİL MODELLERİ İLE GÜVENLİK AÇIĞI ANALİZİ ve GÜVENLİ KODLAMA: LORA TABANLI BİR YAKLAŞIM

## 📖 Proje Özeti
Bu proje, yazılım güvenliği ve kaynak kod analizi alanında Büyük Dil Modellerinin (LLM) performansını artırmayı amaçlamaktadır. Proje kapsamında, `Qwen2.5-Coder-7B-Instruct` modeli taban alınarak, çeşitli siber güvenlik ve zafiyet veri setleri üzerinde LoRA (Low-Rank Adaptation) tekniği ile ince ayar (fine-tuning) yapılmıştır. Eğitilen modelin ("Fenrir" ve diğer türevleri) zafiyet tespiti, saldırı vektörü analizi ve güvenli kodlama önerileri sunma yetenekleri geliştirilmiştir.

Eğitim süreçlerinin yanı sıra, projenin en önemli parçalarından biri Gradio tabanlı **Siber Güvenlik LLM Test ve Sohbet Arayüzü**'dür. Bu arayüz, araştırmacıların eğitilmiş modelleri ve temel (eğitilmemiş) modelleri kıyaslamasına, veri setlerinden gerçek zafiyet örnekleri çekerek model yanıtlarını ROUGE ve Semantik Benzerlik (Semantic Similarity) metrikleriyle otomatik olarak değerlendirmesine olanak tanır.

## ✨ Temel Özellikler
- **LoRA ile Verimli İnce Ayar (Fine-Tuning):** Unsloth kütüphanesi kullanılarak düşük donanım gereksinimleriyle (4-bit kuantalama vb.) hızlı ve verimli SFT (Supervised Fine-Tuning) işlemi.
- **Çoklu Veri Seti Desteği:** BigVul, Fenrir, SecureCodeWeb ve Mega-Vul gibi önde gelen siber güvenlik veri setleriyle entegrasyon.
- **Otomatik Veri Hazırlama:** Ham veri setlerini Qwen'in beklediği ChatML formatına (`<|im_start|>system/user/assistant<|im_end|>`) dönüştüren ve token sınırlarını (örn. 1024 token) aşan verileri temizleyen araçlar.
- **Gelişmiş Web Arayüzü (`ChatWebSite.py`):** 
  - Model seçimi (Eğitilmiş Özel Model vs Temel Qwen2.5 Modeli).
  - İlgili veri setinden "Zafiyetli", "Zafiyetsiz" veya "Rastgele" kod bloklarını prompt olarak getirme.
  - Modelin ürettiği çıktıyı "Ground Truth" (beklenen orjinal analiz yanıtı) ile karşılaştırma.
  - ROUGE-1, ROUGE-2, ROUGE-L ve Sentence Transformers tabanlı Cosine Similarity ile otomatik metrik hesaplama.
- **Merkezi Yapılandırma:** Tüm veri seti ve model dosya yollarının tek bir `config.json` dosyası üzerinden dinamik olarak kolayca yönetilmesi.

## 📂 Klasör ve Dosya Yapısı

```text
├── ChatWebSite.py              # Gradio tabanlı test, sohbet ve değerlendirme arayüzü
├── config.json                 # Dosya yollarının ve model konumlarının tutulduğu konfigürasyon dosyası
├── Training/                   # Model Eğitim (Fine-tuning) Betikleri (Unsloth Tabanlı)
│   ├── BigVul.py               # BigVul veri seti ile eğitim
│   ├── CodeSearchNet.py        # CodeSearchNet veri seti ile eğitim
│   ├── Fenrir.py               # Fenrir v2.1 veri seti ile eğitim
│   └── SecureCodeWeb.py        # SecureCodeWeb veri seti ile eğitim
├── Dataset/                    # Veri Seti İşleme ve Hazırlık Betikleri
│   ├── BigVul/
│   │   ├── BigVulBalance.py    # Veri dengesizliğini giderme ve birleştirme
│   │   └── BigVulFilter.py     # Tokenizasyon, ChatML formatlama ve uzun metinleri filtreleme
│   └── SecureCodeWeb/
│       └── SecureCodeWebDatasetPrepare.py # ShareGPT formatını ChatML formatına dönüştürme
└── README.md                   # Bu dosya (Proje Dokümantasyonu)
```

## 🛠️ Kurulum ve Gereksinimler

Projeyi yerel makinenizde veya sunucunuzda çalıştırmak için Python 3.8 ve üzeri önerilmektedir. Temel kütüphaneleri yüklemek için:

```bash
# Model Eğitimi İçin Gerekli Olanlar
pip install torch transformers datasets accelerate bitsandbytes peft trl

# Unsloth Kurulumu (Donanıma ve CUDA sürümüne göre farklılık gösterebilir)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Web Arayüzü ve Metrik Hesaplama İçin Gerekli Olanlar
pip install gradio sentence-transformers rouge-score evaluate scikit-learn
```

> **Not:** Eğitim scriptleri (`Training/` altındakiler) varsayılan olarak Google Colab ortamı ve Google Drive entegrasyonu düşünülerek kurgulanmıştır. Ancak, entegre edilen `config.json` yapısı sayesinde ilgili veri setlerini ve çıktı klasörlerini kendi yerel dizinlerinize ayarlayarak Jupyter Notebook veya terminal üzerinden yerelde de çalıştırabilirsiniz.

## 🚀 Kullanım

### 1. Yapılandırma (Config)
Proje kök dizinindeki `config.json` dosyasını açarak yerel makinenizdeki veya bulut ortamınızdaki model, veri seti ve checkpoint yollarını kendi ortamınıza göre güncelleyin. Bu sayede kod içerisinde hiçbir `D:\` veya `/content/drive/...` yolunu manuel değiştirmek zorunda kalmazsınız.

### 2. Veri Seti Hazırlığı
İndirdiğiniz veya kullanmak istediğiniz veri setlerini eğitmeden önce modele uygun formata sokmak için `Dataset/` klasörü altındaki betikleri kullanabilirsiniz.

Örneğin, BigVul veri setini hazırlamak için sırasıyla:
```bash
python Dataset/BigVul/BigVulFilter.py
python Dataset/BigVul/BigVulBalance.py
```

### 3. Model Eğitimi (Fine-Tuning)
Eğitim işlemleri için `Training/` altındaki betikleri kullanın. Bu scriptler, `unsloth` kullanarak bellek dostu bir eğitim süreci başlatır.

Örnek kullanım (SecureCodeWeb veri seti için eğitim başlatma):
```bash
python Training/SecureCodeWeb.py
```

### 4. Web Arayüzü (Gradio ile Test ve Analiz)
Eğitilmiş modelinizi gerçek zamanlı test etmek, sohbet etmek ve model başarım metriklerini değerlendirmek için arayüzü başlatın:
```bash
python ChatWebSite.py
```

- Tarayıcınızda `http://localhost:7860` (veya terminalde belirtilen IP) adresine giderek uygulamaya erişebilirsiniz.
- **Model Yükle:** Uygulama içindeki menüden `config.json`'da belirttiğiniz Eğitilmiş Özel Modeli veya Temel Qwen Modelini yükleyebilirsiniz.
- **Veri Getir:** İstediğiniz veri setini seçip **Veri Getir** butonuna tıklayarak modele sunulacak zafiyetli kodu yükleyin. (Aynı zamanda modelin bu koda vermesi beklenen orjinal doğru yanıt arka planda "Ground Truth" olarak kaydedilir).
- **Gönder:** Kod analizini başlatın. Çıktı tamamlandığında sistem, ROUGE ve Semantik Benzerlik skorlarını otomatik hesaplayarak metrikler panelinde gösterir.

## 📊 Değerlendirme Metrikleri
Modelin siber güvenlik bağlamındaki muhakeme (reasoning) yeteneğini ve açıklama kalitesini ölçmek için arayüze aşağıdaki iki metrik entegre edilmiştir:

1. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** 
   ROUGE-1 (kelime), ROUGE-2 (ikili kelime) ve ROUGE-L (en uzun ortak alt dizi) skorları ile modelin ürettiği çıktının, orjinal insan/uzman yanıtı ile sözdizimsel N-gram örtüşmesini ölçer.
2. **Semantik Benzerlik (Semantic Similarity / Cosine Similarity):** 
   `sentence-transformers/all-MiniLM-L6-v2` modeli kullanılarak üretilen metin ve beklenen metin vektörel (embedding) forma dönüştürülür ve aralarındaki kosinüs benzerliği hesaplanır. Siber güvenlik jargonunda modelin kelimesi kelimesine aynı cümleyi kurmasa bile **aynı anlamsal sonuca (örn. zafiyet tipine ve çözümüne)** ulaşıp ulaşmadığını ölçmede kritik öneme sahiptir.

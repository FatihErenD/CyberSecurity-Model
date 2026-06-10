from datasets import load_dataset, concatenate_datasets

# 1. Dosyaları ayrı ayrı yükle
print("Eğitim (Train) dosyası yükleniyor...")
train_dataset = load_dataset("json", data_files="bigvul_train_balanced_1_to_2.jsonl", split="train")

print("Doğrulama (Val) dosyası yükleniyor...")
val_dataset = load_dataset("json", data_files="bigvul_val_balanced_1_to_2.jsonl", split="train")

# 2. Veri setlerini birleştir
print("\nVeri setleri birleştiriliyor...")
combined_dataset = concatenate_datasets([train_dataset, val_dataset])

# 3. Modeli kafa karışıklığından korumak için veriyi iyice karıştır
print("Birleştirilmiş veri seti karıştırılıyor (shuffle)...")
combined_dataset = combined_dataset.shuffle(seed=42)

# 4. Yeni dosyayı diske kaydet
output_file = "bigvul_combined_balanced_1_to_2.jsonl"
print("Dosya diske yazılıyor...")
combined_dataset.to_json(output_file, orient="records", lines=True)

# Sonuç Raporu
print(f"\n✅ İşlem Başarıyla Tamamlandı!")
print(f"Eğitim Verisi Boyutu   : {len(train_dataset)}")
print(f"Doğrulama Verisi Boyutu: {len(val_dataset)}")
print("-" * 35)
print(f"Yeni Toplam Veri Boyutu: {len(combined_dataset)}")
print(f"Birleştirilmiş dosya '{output_file}' olarak kaydedildi.")
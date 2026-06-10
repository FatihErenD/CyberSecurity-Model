import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Modelin Kendi Tokenizer'ını Yüklüyoruz
model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def format_and_count_tokens(example):
    """
    Hem formatlama yapar hem de Qwen'in gözünden token sayısını hesaplar.
    """
    code = str(example.get('func_before', ''))
    is_vul = example.get('vul', 0)
    cwe_id = str(example.get('CWE ID', example.get('cwe_id', '')))

    system_prompt = "Sen siber güvenlik alanında uzmanlaşmış, statik kod analizi yapan bir yapay zeka asistanısın."
    user_prompt = (
        "Aşağıdaki C/C++ kodunu analiz et. Kodda herhangi bir güvenlik zafiyeti olup olmadığını "
        "ve varsa bunun hangi zafiyet türüne (CWE) ait olduğunu belirle.\n\n"
        f"Kod:\n{code}"
    )

    if is_vul == 1:
        cwe_label = cwe_id if cwe_id and "CWE" in cwe_id else "Bilinmeyen CWE"
        assistant_response = f"Zafiyet Durumu: Zafiyetli\nZafiyet Türü: {cwe_label}"
    else:
        assistant_response = "Zafiyet Durumu: Güvenli\nZafiyet Türü: Yok"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
    ]

    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    token_count = len(tokenizer(chat_text)["input_ids"])

    return {"messages": messages, "token_count": token_count}


# ==========================================
# WINDOWS İÇİN KRİTİK KORUMA BLOĞU
# ==========================================
if __name__ == '__main__':
    print(f"'{model_id}' tokenizer'ı yüklendi.")
    print("Hugging Face üzerinden BigVul veri seti indiriliyor...")

    dataset = load_dataset("bstee615/bigvul", split="validation")

    print("Veri seti formatlanıyor ve token sayıları hesaplanıyor (Bu işlem biraz sürebilir)...")

    processed_dataset = dataset.map(
        format_and_count_tokens,
        remove_columns=dataset.column_names,
        num_proc=4  # Artık Windows'ta çökmeden çalışacak
    )

    print("\n--- Veri Seti Token İstatistikleri ---")
    token_counts = processed_dataset["token_count"]
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    max_tokens = np.max(token_counts)
    min_tokens = np.min(token_counts)

    print(f"Toplam Örnek Sayısı: {len(token_counts)}")
    print(f"Ortalama Token: {mean_tokens:.2f}")
    print(f"Medyan Token: {median_tokens:.2f}")
    print(f"En Kısa Örnek: {min_tokens} token")
    print(f"En Uzun Örnek: {max_tokens} token")
    print("--------------------------------------\n")

    MAX_TOKEN_LIMIT = 1024
    print(f"{MAX_TOKEN_LIMIT} token sınırını aşan örnekler temizleniyor...")

    filtered_dataset = processed_dataset.filter(lambda x: x["token_count"] <= MAX_TOKEN_LIMIT)
    removed_count = len(processed_dataset) - len(filtered_dataset)

    print(f"Silinen örnek sayısı: {removed_count}")
    print(f"Kalan temiz örnek sayısı: {len(filtered_dataset)}")

    filtered_dataset = filtered_dataset.remove_columns(["token_count"])

    print("\nVeri seti %90 Eğitim ve %10 Test olarak bölünüyor...")
    split_dataset = filtered_dataset

    print("Dosyalar diske kaydediliyor...")
    # split_dataset.to_json("bigvul_train_filtered.jsonl", orient="records", lines=True)
    split_dataset.to_json("bigvul_val_filtered.jsonl", orient="records", lines=True)
    # split_dataset.to_json("bigvul_test_filtered.jsonl", orient="records", lines=True)

    print("İşlem başarıyla tamamlandı! Eğitime hazır, güvenli veri seti oluşturuldu.")
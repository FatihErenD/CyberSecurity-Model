import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def format_and_tokenize(example, tokenizer, system_message):
    # 1. Sütun ismini dinamik olarak yakala
    if 'conversations' in example:
        messages = example['conversations']
    elif 'messages' in example:
        messages = example['messages']
    else:
        raise KeyError(f"Konuşma verisi bulunamadı! Veri setindeki mevcut sütunlar: {list(example.keys())}")

    # 2. Uyumluluk Kontrolü: Eski ShareGPT formatıysa (from/value) Qwen'in beklediği role/content formatına çevir
    if len(messages) > 0 and 'from' in messages[0]:
        role_mapping = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = [{"role": role_mapping.get(msg["from"], msg["from"]), "content": msg["value"]} for msg in messages]

    # 3. System prompt ekleme
    if messages[0]['role'] != 'system':
        messages = [system_message] + messages

    # 4. Qwen formatına (ChatML) çevirme
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # 5. Token uzunluğu hesaplama
    tokens = tokenizer(text, add_special_tokens=False)

    return {
        "text": text,
        "token_length": len(tokens["input_ids"])
    }


if __name__ == '__main__':
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    print(f"{model_id} tokenizer'ı yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("SecureCode Web veri seti yükleniyor...")
    dataset = load_dataset("scthornton/securecode-web", split="train")

    system_message = {
        "role": "system",
        "content": "Sen kıdemli bir uygulama güvenliği mühendisisin. Görevin, sağlanan koddaki potansiyel zafiyetleri OWASP Top 10 standartlarına göre analiz etmek, saldırı vektörlerini (exploit) açıklamak ve ardından güvenli uygulamayı derinlemesine savunma (defense-in-depth) stratejileriyle birlikte sunmaktır."
    }

    print("Veri seti Qwen formatına çevriliyor ve token sayıları hesaplanıyor...")

    processed_dataset = dataset.map(
        format_and_tokenize,
        fn_kwargs={"tokenizer": tokenizer, "system_message": system_message},
        num_proc=4,
        desc="Formatlama ve Token Sayımı"
    )

    lengths = processed_dataset["token_length"]

    print("\n" + "=" * 40)
    print("📊 TOKEN İSTATİSTİKLERİ")
    print("=" * 40)
    print(f"Toplam Örnek Sayısı : {len(lengths)}")
    print(f"Minimum Token       : {np.min(lengths)}")
    print(f"Maksimum Token      : {np.max(lengths)}")
    print(f"Ortalama Token      : {np.mean(lengths):.2f}")
    print(f"Medyan Token        : {np.median(lengths)}")
    print(f"%95'lik Dilim (P95) : {np.percentile(lengths, 95):.0f}")
    print("=" * 40 + "\n")

    output_dir = "./securecode_web_qwen_processed"
    processed_dataset.save_to_disk(output_dir)
    print(f"✅ İşlenmiş veri seti '{output_dir}' dizinine başarıyla kaydedildi.")
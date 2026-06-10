import gradio as gr
import json
import random
import os
import torch
from datasets import load_dataset
from torch_geometric.llm.models.txt2kg import SYSTEM_PROMPT
from unsloth import FastLanguageModel
import threading

# Metrics setup
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

st_model = None
st_model_loaded = False

def get_st_model():
    global st_model, st_model_loaded
    if not st_model_loaded:
        try:
            from sentence_transformers import SentenceTransformer, util
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            st_model = None
        st_model_loaded = True
    return st_model

def calculate_metrics(generated_text, expected_text):
    if not expected_text or expected_text.strip() == "":
        return ""

    metrics_str = "### 📊 Değerlendirme Metrikleri\n"

    # ROUGE
    if rouge_scorer:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(expected_text, generated_text)
        metrics_str += f"- **ROUGE-1:** {scores['rouge1'].fmeasure:.4f}\n"
        metrics_str += f"- **ROUGE-2:** {scores['rouge2'].fmeasure:.4f}\n"
        metrics_str += f"- **ROUGE-L:** {scores['rougeL'].fmeasure:.4f}\n"
    else:
        metrics_str += "- *ROUGE Metrikleri: `rouge-score` kütüphanesi eksik.* Lütfen `pip install rouge-score` komutunu çalıştırın.\n"
        
    # Semantic Similarity
    model_st = get_st_model()
    if model_st:
        try:
            from sentence_transformers import util
            emb1 = model_st.encode(expected_text, convert_to_tensor=True)
            emb2 = model_st.encode(generated_text, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(emb1, emb2).item()
            metrics_str += f"- **Semantik Benzerlik:** {sim:.4f}\n"
        except Exception as e:
            metrics_str += f"- *Semantik Benzerlik hesaplanamadı:* {str(e)}\n"
    else:
        metrics_str += "- *Semantik Benzerlik: `sentence-transformers` kütüphanesi eksik.* Lütfen `pip install sentence-transformers` komutunu çalıştırın.\n"

    return metrics_str


# ========================================
# GLOBAL VARIABLES
# ========================================
model = None
tokenizer = None
MAX_SEQ_LENGTH = 32768

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
        chat_config = config.get("ChatWebSite", {})
except Exception as e:
    print(f"Config yüklenirken hata oluştu: {e}")
    chat_config = {}

# Dataset Paths
BIGVUL_PATH = chat_config.get("BIGVUL_PATH")
MEGAVUL_PATH = chat_config.get("MEGAVUL_PATH")


SYSTEM_PROMPT = """
You are an expert cybersecurity analyst specializing in:

- Threat modeling
- Vulnerability analysis
- Secure software architecture
- Incident response
- Threat intelligence
- Adversarial reasoning
- Blockchain and smart contract security
- Cloud, application, network, identity, and infrastructure security

Your primary objective is to provide technically accurate, evidence-based security analysis.

Core Principles:

1. Accuracy is more important than completeness.
2. Do not invent CVEs, threat intelligence, MITRE mappings, attack stages, risk scores, or incidents.
3. If evidence is unavailable, explicitly state the limitation.
4. Only include analytical sections that are relevant to the specific security topic.
5. Prefer concrete attack mechanics over generic security language.
6. Distinguish clearly between:
   - Observed attacks
   - Documented research
   - Theoretical risks
7. Explain causal relationships, not just descriptions.
8. Focus on how and why an attack works.
9. When discussing defenses, explain which attack step they mitigate.
10. If multiple attack paths exist, compare them.

Response Structure (Adaptive)

## Direct Answer
Provide a concise answer to the security question.

## Technical Analysis
Explain the vulnerability, attack technique, security mechanism, or threat.

## Attack Mechanics
Describe exploitation paths, attacker capabilities, prerequisites, and affected assets.

## Security Impact
Explain technical, operational, and business consequences.

## Detection Opportunities
Describe indicators, monitoring opportunities, behavioral anomalies, and telemetry.

## Mitigations
Provide preventive, detective, and response controls.

## Threat Assessment
Assess:
- Likelihood
- Impact
- Required attacker capability
- Exploit complexity

Optional Sections
Only include when relevant:

- MITRE ATT&CK Mapping
- CVEs
- Threat Intelligence
- Incident Examples
- Kill Chain Analysis
- Supply Chain Risks
- Cloud-Specific Risks
- Blockchain-Specific Risks
- AI/LLM Risks

Important:

Do not force a topic into a kill chain if no kill chain exists.
Do not force MITRE ATT&CK mappings.
Do not force CVSS scoring.
Do not generate placeholder text.
Omit irrelevant sections entirely.

The goal is expert-level cybersecurity reasoning rather than template completion.
"""

MODELS = {
    "Eğitilmiş Model (Fenrir)": chat_config.get("MODELS_FENRIR"),
    "Eğitilmemiş Model (Qwen2.5-Coder-7B-Instruct)": chat_config.get("MODELS_QWEN")
}

DEFAULT_MODEL_PATH = chat_config.get("DEFAULT_MODEL_PATH")

# ========================================
# MODEL LOADING
# ========================================
def load_model(selected_model_name):
    global model, tokenizer

    model_path = MODELS.get(selected_model_name, DEFAULT_MODEL_PATH)

    try:
        print(f"Yükleniyor: {model_path}")

        if model is not None:
            del model
            del tokenizer
            torch.cuda.empty_cache()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.config.use_cache = True

        return f"{selected_model_name} başarıyla yüklendi!"

    except Exception as e:
        return f"Model yüklenirken hata oluştu: {e}"
# ========================================
# DATASET FETCHING
# ========================================
def fetch_bigvul(filter_type):
    try:
        with open(BIGVUL_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        random.shuffle(lines)
        for line in lines:
            data = json.loads(line)
            msgs = data.get("messages", [])
            if not msgs: continue

            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            ast_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

            is_vuln = "zafiyetli" in ast_msg.lower()

            if filter_type == "Zafiyetli" and not is_vuln: continue
            if filter_type == "Zafiyetsiz" and is_vuln: continue

            return user_msg, ast_msg

    except Exception as e:
        return f"Hata: {str(e)}", ""
    return "Uygun veri bulunamadı.", ""


def fetch_megavul(filter_type):
    try:
        with open(MEGAVUL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        random.shuffle(data)
        for item in data:
            is_vul = int(bool(item.get("is_vul", 0)))
            code = item.get("abstract_func") or item.get("func", "")

            if filter_type == "Zafiyetli" and is_vul != 1: continue
            if filter_type == "Zafiyetsiz" and is_vul != 0: continue

            # Mega-Vul is binary, so simulate an assistant response
            ast_msg = "Zafiyet Durumu: Zafiyetli\nZafiyet Türü: Belirsiz" if is_vul == 1 else "Zafiyet Durumu: Güvenli\nZafiyet Türü: Yok"
            return code, ast_msg

    except Exception as e:
        return f"Hata: {str(e)}", ""
    return "Uygun veri bulunamadı.", ""


def fetch_fenrir(filter_type):
    try:
        dataset = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1", split="train")
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        security_keywords = ["vulnerability", "cve", "injection", "xss", "owasp", "exploit", "patch", "secure"]

        for idx in indices:
            item = dataset[idx]
            user_text = item.get("user", "")
            ast_text = item.get("assistant", "")

            is_vuln = any(kw in ast_text.lower() for kw in security_keywords)

            if filter_type == "Zafiyetli" and not is_vuln: continue
            if filter_type == "Zafiyetsiz" and is_vuln: continue

            return user_text, ast_text

    except Exception as e:
        return f"Hata: {str(e)}", ""
    return "Uygun veri bulunamadı.", ""


def fetch_securecodeweb(filter_type):
    try:
        dataset = load_dataset("scthornton/securecode-web", split="test")
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        security_keywords = ["vulnerability", "cve", "injection", "xss", "owasp", "exploit", "patch", "secure"]

        for idx in indices:
            item = dataset[idx]
            msgs = item.get('messages', item.get('conversations', []))

            user_text = ""
            ast_text = ""
            for msg in msgs:
                r = msg.get('role', msg.get('from', '')).lower()
                c = msg.get('content', msg.get('value', ''))
                if r in ['human', 'user']:
                    user_text = c
                elif r in ['gpt', 'assistant', 'model', 'bot']:
                    ast_text = c

            is_vuln = any(kw in ast_text.lower() for kw in security_keywords)

            if filter_type == "Zafiyetli" and not is_vuln: continue
            if filter_type == "Zafiyetsiz" and is_vuln: continue

            if user_text:
                return user_text, ast_text

    except Exception as e:
        return f"Hata: {str(e)}", ""
    return "Uygun veri bulunamadı.", ""


def set_prompt(dataset_choice, filter_choice):
    user_txt, expected_txt = "", ""
    if dataset_choice == "BigVul":
        user_txt, expected_txt = fetch_bigvul(filter_choice)
    elif dataset_choice == "Fenrir":
        user_txt, expected_txt = fetch_fenrir(filter_choice)
    elif dataset_choice == "SecureCodeWeb":
        user_txt, expected_txt = fetch_securecodeweb(filter_choice)
    elif dataset_choice == "Mega-Vul":
        user_txt, expected_txt = fetch_megavul(filter_choice)

    return user_txt, expected_txt


# ========================================
# CHAT LOGIC
# ========================================
def custom_chat(user_message, history, expected_answer=""):
    global model, tokenizer

    if history is None:
        history = []

    if model is None or tokenizer is None:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "Lütfen önce bir model yükleyin!"})
        yield history, "### 📊 Değerlendirme Metrikleri\n*(Model yüklü değil)*", ""
        return

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""})
    
    messages = []

    # System prompt
    if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })

    # Sohbet geçmişi (Gradio'dan dönen format dict veya obje olabilir)
    for item in history[:-2]:
        if isinstance(item, dict):
            messages.append(item)
        elif hasattr(item, "role") and hasattr(item, "content"):
            messages.append({"role": item.role, "content": item.content})
        elif isinstance(item, (list, tuple)):
            messages.append({
                "role": "user",
                "content": str(item[0])
            })
            if item[1]:
                messages.append({
                    "role": "assistant",
                    "content": str(item[1])
                })

    # Kullanıcının son mesajı
    messages.append({
        "role": "user",
        "content": str(user_message)
    })

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH
    ).to(model.device)

    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )
    thread.start()

    generated_text = ""

    for new_text in streamer:
        generated_text += new_text
        if isinstance(history[-1], dict):
            history[-1]["content"] = generated_text.replace("\\n", "\n")
        else:
            history[-1].content = generated_text.replace("\\n", "\n")
        yield history, "### 📊 Değerlendirme Metrikleri\n*(Cevap oluşturuluyor...)*", ""
        
    # Metrikleri hesapla ve sonuca ekle
    metrics_output = calculate_metrics(generated_text, expected_answer)
    if not metrics_output:
        metrics_output = "### 📊 Değerlendirme Metrikleri\n*(Ground truth bulunamadı veya boş)*"
        
    yield history, metrics_output, ""

# ========================================
# GRADIO UI
# ========================================
with gr.Blocks(title="CyberSecurity LLM Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Siber Güvenlik LLM Test ve Sohbet Arayüzü")

    with gr.Row():
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=["Eğitilmiş Model (Fenrir)", "Eğitilmemiş Model (Qwen2.5-Coder-7B-Instruct)"],
                value="Eğitilmiş Model (Fenrir)",
                label="Kullanılacak Modeli Seçin"
            )
        with gr.Column(scale=1):
            load_btn = gr.Button("🚀 Modeli Yükle", variant="primary")

    status_text = gr.Textbox(label="Durum", interactive=False)
    load_btn.click(fn=load_model, inputs=[model_dropdown], outputs=[status_text])

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Veri Setinden Prompt Çek")
            dataset_dropdown = gr.Dropdown(
                choices=["BigVul", "Fenrir", "SecureCodeWeb", "Mega-Vul"],
                value="BigVul",
                label="Veri Seti"
            )
            filter_dropdown = gr.Dropdown(
                choices=["Rastgele", "Zafiyetli", "Zafiyetsiz"],
                value="Rastgele",
                label="Zafiyet Durumu (Filtre)"
            )
            fetch_btn = gr.Button("📦 Veri Getir")
            gr.Markdown(
                "⚠️ **ÖNEMLİ:** Farklı bir veri setinden yeni kod getirdiğinizde, modelin önceki kodlarla yenisini karıştırmaması için lütfen sohbet penceresindeki **🗑️ Clear (Çöp Kutusu)** butonuna basarak hafızayı temizleyin!")

            gr.Markdown("**Beklenen Orijinal Yanıt (Ground Truth):**")
            expected_answer_box = gr.Textbox(label="", lines=6, interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Model İle Sohbet")

            custom_chatbot = gr.Chatbot(height=500)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="Mesajınızı buraya yazın...",
                    container=False
                )
                submit_btn = gr.Button("🚀 Gönder", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Sohbeti Temizle")

            gr.Markdown("---")
            metrics_box = gr.Markdown("### 📊 Değerlendirme Metrikleri\n*(Henüz hesaplanmadı)*")
            
            # Action Bindings
            submit_event = submit_btn.click(
                fn=custom_chat,
                inputs=[msg_input, custom_chatbot, expected_answer_box],
                outputs=[custom_chatbot, metrics_box, msg_input]
            )
            msg_input.submit(
                fn=custom_chat,
                inputs=[msg_input, custom_chatbot, expected_answer_box],
                outputs=[custom_chatbot, metrics_box, msg_input]
            )
            
            clear_btn.click(lambda: ([], "### 📊 Değerlendirme Metrikleri\n*(Henüz hesaplanmadı)*", ""), None, [custom_chatbot, metrics_box, msg_input], queue=False)

    fetch_btn.click(
        fn=set_prompt,
        inputs=[dataset_dropdown, filter_dropdown],
        outputs=[msg_input, expected_answer_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)

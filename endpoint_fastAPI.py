import os
import json
import requests
import torch
import openai
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration

# --- Cargar variables de entorno ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# --- Inicializar FastAPI ---
app = FastAPI()

# --- Configuración de dispositivos ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TARGET_LEN = 40

# --- Inicializar modelos ---
MODEL_DIR_T5 = "t5_namuywam"
MODEL_DIR_MBART = "fine_tuned_mbart"
MODEL_DIR_NLLB = "fine_tuned_nllb"

models = {}

def load_model(model_path, model_type):
    try:
        if model_type == "t5":
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = T5ForConditionalGeneration.from_pretrained(model_path)
        elif model_type in ["mbart", "nllb"]:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            raise ValueError(f"Modelo no soportado: {model_type}")
        model.to(DEVICE).eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error al cargar {model_type}: {e}")
        return None, None

models["t5"] = load_model(MODEL_DIR_T5, "t5")
models["mbart"] = load_model(MODEL_DIR_MBART, "mbart")
models["nllb"] = load_model(MODEL_DIR_NLLB, "nllb")

# --- Cliente OpenAI ---
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Función de traducción ---
def translate_sentence(sentence: str, model_key: str, direction: str) -> str:
    tokenizer, model = models.get(model_key, (None, None))
    if not tokenizer or not model:
        return f"Error: Modelo '{model_key}' no disponible."
    try:
        if model_key == "t5":
            input_text = f"translate {direction}: {sentence}"
            inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        elif model_key == "mbart":
            tokenizer.src_lang = "unspecified_UNKNOWN" if direction == "nmw-es" else "es_XX"
            tokenizer.tgt_lang = "es_XX" if direction == "nmw-es" else "unspecified_UNKNOWN"
            inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
        elif model_key == "nllb":
            src = "nmw_Latn" if direction == "nmw-es" else "spa_Latn"
            tgt = "spa_Latn" if direction == "nmw-es" else "nmw_Latn"
            inputs = tokenizer(sentence, return_tensors="pt", src_lang=src).to(DEVICE)

        generate_kwargs = {"max_length": MAX_TARGET_LEN}
        if model_key == "nllb":
            generate_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id[tgt]
        elif model_key == "mbart":
            generate_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id[tokenizer.tgt_lang]

        output = model.generate(**inputs, **generate_kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error traduciendo con {model_key}: {e}")
        return "Error en la traducción"

# --- Función para manejar respuesta de OpenAI ---
def get_openai_response(prompt: str) -> dict:
    system_prompt = (
        "Eres un asistente útil y experto en idiomas. Tu tarea es procesar las solicitudes del usuario. "
        "Si el usuario pide una traducción, responde en formato JSON con la frase y la dirección "
        "('nmw-es' o 'es-nmw'). Si es una conversación, responde en texto plano.\n\n"
        '{"action": "translate", "text": "Kukapi", "direction": "nmw-es"}\n'
        '{"action": "translate", "text": "Hola mundo", "direction": "es-nmw"}\n'
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("{") and content.endswith("}"):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        return {"action": "chat", "response": content}
    except Exception as e:
        print(f"Error con OpenAI: {e}")
        return {"action": "error", "response": "Error al contactar OpenAI"}

# --- Enviar mensaje a Telegram ---
def send_message(chat_id, text):
    url = f"{BASE_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, json=payload)
    except requests.exceptions.RequestException as e:
        print(f"Error enviando a Telegram: {e}")

# --- Ruta Webhook ---
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    print(f"Recibido: {data}")

    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"]

        try:
            gpt_parsed = get_openai_response(user_message)
            if gpt_parsed.get("action") == "translate":
                text = gpt_parsed["text"]
                direction = gpt_parsed["direction"]
                translated = translate_sentence(text, "mbart", direction)
                response_text = f"Traducción de '{text}' ({direction}): {translated}"
            elif gpt_parsed.get("action") == "chat":
                response_text = gpt_parsed["response"]
            else:
                response_text = gpt_parsed.get("response", "No pude entender tu solicitud.")
        except Exception as e:
            response_text = f"Ocurrió un error: {e}"
        send_message(chat_id, response_text)

    return {"status": "ok"}

import os
import openai
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
import torch

# Cargar variables de entorno
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hiperparámetros (ajusta si tu modelo usó otros)
MAX_TARGET_LEN = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ruta del modelo
MODEL_DIR_T5 = "t5_namuywam"
MODEL_DIR_MBART = "fine_tuned_mbart"
MODEL_DIR_NLLB = "fine_tuned_nllb"

# Diccionario para almacenar tokenizers y modelos
models = {}

def load_model(model_path, model_type):
    """Carga un modelo y su tokenizer."""
    try:
        if model_type == "t5":
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = T5ForConditionalGeneration.from_pretrained(model_path)
        elif model_type in ["mbart", "nllb"]:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        model.to(DEVICE)
        model.eval() # Poner el modelo en modo evaluación
        print(f"Modelo {model_type.upper()} y tokenizer cargados exitosamente desde {model_path} en {DEVICE}")
        return tokenizer, model
    except Exception as e:
        print(f"Error al cargar el modelo {model_type.upper()} desde {model_path}: {e}")
        return None, None
    
# Cargar todos los modelos
models["t5"] = load_model(MODEL_DIR_T5, "t5")
models["mbart"] = load_model(MODEL_DIR_MBART, "mbart")
models["nllb"] = load_model(MODEL_DIR_NLLB, "nllb")

# Configuración de OpenAI
client = openai.OpenAI(api_key=OPENAI_API_KEY)
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def translate_sentence(sentence: str, model_key: str, direction: str) -> str:
    """
    Traduce una oración usando el modelo especificado en la dirección dada.
    Args:
        sentence (str): La oración a traducir.
        model_key (str): 't5', 'mbart', o 'nllb'.
        direction (str): 'nmw-es' para Namuy-wam a Español, 'es-nmw' para Español a Namuy-wam.
    """
    tokenizer, model = models.get(model_key, (None, None))
    if not tokenizer or not model:
        return f"Error: Modelo '{model_key}' no disponible."

    try:
        if model_key == "t5":
            input_text = f"translate {direction}: {sentence}"
            inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        elif model_key == "mbart":
            # Para mBART, necesitamos configurar src_lang y tgt_lang en el tokenizer
            if direction == "nmw-es":
                tokenizer.src_lang = "unspecified_UNKNOWN" # O el código ISO real de Namuy-wam
                tokenizer.tgt_lang = "es_XX"
            elif direction == "es-nmw":
                tokenizer.src_lang = "es_XX"
                tokenizer.tgt_lang = "unspecified_UNKNOWN" # O el código ISO real de Namuy-wam
            
            inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
        elif model_key == "nllb":
            # Para NLLB, se especifica src_lang y tgt_lang directamente en la llamada a tokenizer
            src_lang_nllb = "nmw_Latn" if direction == "nmw-es" else "spa_Latn"
            tgt_lang_nllb = "spa_Latn" if direction == "nmw-es" else "nmw_Latn"
            
            inputs = tokenizer(sentence, return_tensors="pt", src_lang=src_lang_nllb).to(DEVICE)
        
        # Generación común para todos los modelos Seq2Seq
        # Para NLLB, forced_bos_token_id asegura que el primer token del decoder sea el token de destino
        # Esto es crucial para la traducción multilingüe con NLLB
        generate_kwargs = {"max_length": MAX_TARGET_LEN}
        if model_key == "nllb":
            if direction == "nmw-es":
                generate_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["spa_Latn"]
            elif direction == "es-nmw":
                generate_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["nmw_Latn"] # Asumiendo que 'nmw_Latn' se añadió/se mapeó
        
        # También para mBART, aunque `tgt_lang` en tokenizer ya lo maneja
        elif model_key == "mbart":
            if direction == "nmw-es":
                generate_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["es_XX"]
            elif direction == "es-nmw":
                generate_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["unspecified_UNKNOWN"] # Asegúrate de que este token exista si lo usas

        output = model.generate(**inputs, **generate_kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error durante la traducción con {model_key} ({direction}): {e}")
        return "Error en la traducción"

def get_openai_response(prompt: str) -> dict:
    """Llama a GPT-4o Mini para identificar la intención del usuario
    (conversación general o solicitud de traducción) y extraer la frase a traducir."""
    system_prompt = (
        "Eres un asistente útil y experto en idiomas. Tu tarea es procesar las solicitudes del usuario. "
        "Si el usuario pide una traducción, DEBES responder en un formato JSON específico que indique "
        "la frase a traducir y la dirección de la traducción. "
        "Las direcciones posibles son 'nmw-es' (Namuy-wam a Español) o 'es-nmw' (Español a Namuy-wam). "
        "Si el usuario no pide una traducción, responde de manera conversacional en texto plano. "
        "\n\nEjemplos de JSON para traducción:\n"
        '{"action": "translate", "text": "Kukapi", "direction": "nmw-es"}\n'
        '{"action": "translate", "text": "Hola mundo", "direction": "es-nmw"}\n'
        "\nEjemplo de texto plano para conversación:\n"
        "¡Claro! ¿En qué más puedo ayudarte hoy?"
        "\n\nNotas Importantes:\n"
        "1. No añadas explicaciones adicionales en la respuesta JSON. SOLO el JSON."
        "2. Identifica el idioma de origen de la frase a traducir para determinar la dirección."
        "3. Si no estás seguro de la dirección, asume 'nmw-es' si parece Namuy-wam."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0, # Bajo -> respuestas más consistentes
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        
        # Parsear como JSON
        if content.startswith('{') and content.endswith('}'):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        
        return {"action": "chat", "response": content}
    
    except openai.APIError as e:
        print(f"Error en la API de OpenAI: {e}")
        return {"action": "error", "response": "Lo siento, tuve un problema al contactar a OpenAI."}
    except Exception as e:
        print(f"Error inesperado al obtener respuesta de OpenAI: {e}")
        return {"action": "error", "response": "Lo siento, ocurrió un error inesperado."}

# --- Servidor Flask ---
app = Flask(__name__)

def send_message(chat_id, text):
    """Envía un mensaje a Telegram."""
    url = f"{BASE_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Mensaje enviado a {chat_id}: {text}")
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar mensaje a Telegram ({chat_id}): {e}")

@app.route("/webhook", methods=["POST"])
def webhook():
    """Recibe mensajes de Telegram y responde con OpenAI o traducción."""
    data = request.get_json()
    print(f"Datos recibidos en webhook: {data}")

    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"]
        print(f"Mensaje de usuario ({chat_id}): {user_message}")

        try:
            gpt_parsed_response = get_openai_response(user_message)
            print(f"Respuesta de GPT: {gpt_response}")

            final_response = ""
            if gpt_parsed_response.get("action") == "translate":
                text_to_translate = gpt_parsed_response["text"]
                direction_to_translate = gpt_parsed_response["direction"]
                translated_text = translate_sentence(text_to_translate, "mbart", direction_to_translate)
                final_response = f"Traducción de '{text_to_translate}' ({direction_to_translate}): {translated_text}"

            elif gpt_parsed_response.get("action") == "chat":
                final_response = gpt_parsed_response["response"]

            else: # Error o acción desconocida
                final_response = gpt_parsed_response.get("response", "Lo siento, no pude procesar tu solicitud.")

        except Exception as e:
            print(f"Excepción en el handler del webhook: {e}")
            final_response = f"Ocurrió un error al procesar tu solicitud: {str(e)}"

        send_message(chat_id, final_response)

    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    import json 
    print("Iniciando servidor Flask en 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
import os
import openai
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Cargar variables de entorno
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hiperparámetros (ajusta si tu modelo usó otros)
MAX_TARGET_LEN = 40
# Ruta del modelo
MODEL_DIR = "t5_namuywam"

try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    print(f"Modelo T5 y tokenizer cargados exitosamente desde {MODEL_DIR}")
except Exception as e:
    print(f"Error al cargar el modelo T5 o el tokenizer desde {MODEL_DIR}: {e}")
    # exit()

# Configuración de OpenAI
client = openai.OpenAI(api_key=OPENAI_API_KEY)
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def translate(sentence):
    """Traduce una oración usando el modelo T5 local."""
    try:
        input_text = f"traduce: {sentence}"
        inputs = tokenizer(input_text, return_tensors="pt")
        output = model.generate(**inputs, max_length=MAX_TARGET_LEN)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error durante la traducción local: {e}")
        return "Error en la traducción"

def es_frase_corta(gpt_response: str) -> bool:
    """Determina si la respuesta GPT parece una frase en namuy-wam para traducir."""
    palabras = gpt_response.strip().split()
    return 1 <= len(palabras) <= 4 and not any(
        palabra in gpt_response.lower()
        for palabra in ["significa", "traduce", "es", "palabra", "traducción", "en español"]
    )

def get_openai_response(prompt):
    """Llama a GPT-4o Mini y le informa que puede usar un traductor personalizado."""
    system_prompt = (
        "Eres un asistente útil. Si el usuario pide traducir algo del namuy-wam al español, "
        "responde solo con la palabra o frase que desea traducir, sin explicaciones. "
        "De lo contrario, responde normalmente."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        print(f"Error en la API de OpenAI: {e}")
        return "Lo siento, tuve un problema al contactar a OpenAI."
    except Exception as e:
        print(f"Error inesperado al obtener respuesta de OpenAI: {e}")
        return "Lo siento, ocurrió un error inesperado."

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
            gpt_response = get_openai_response(user_message)
            print(f"Respuesta de GPT: {gpt_response}")

            # Traducir si es una frase corta en namuy-wam
            if es_frase_corta(gpt_response):
                print(f"Intentando traducir con T5: '{gpt_response}'")
                translated = translate(gpt_response)
                final_response = f"Traducción: {translated}"
            else:
                final_response = gpt_response
        except Exception as e:
            print(f"Excepción en el handler del webhook: {e}")
            final_response = f"Ocurrió un error al procesar tu solicitud: {str(e)}"

        send_message(chat_id, final_response)

    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    print("Iniciando servidor Flask en 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
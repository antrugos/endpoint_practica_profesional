#!/bin/bash

# 1. Cargar variables de entorno desde .env
export $(grep -v '^#' .env | xargs)

# 2. (Opcional) Activar entorno virtual
# source venv/bin/activate

# 3. Ejecutar Flask app en segundo plano
echo "Iniciando servidor Flask en segundo plano..."
python endpoint.py &

# Guardar PID para detener Flask después
FLASK_PID=$!

# 4. Iniciar ngrok (requiere que esté instalado y autenticado)
echo "Iniciando túnel ngrok en el puerto 5000..."
./ngrok http 5000 > /dev/null &

# Esperar unos segundos a que ngrok levante y registre el túnel
sleep 5

# 5. Obtener la URL pública de ngrok
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o 'https://[a-zA-Z0-9.-]*\.ngrok\.io' | head -n 1)

if [[ -z "$NGROK_URL" ]]; then
  echo "Error: no se pudo obtener la URL de ngrok."
  kill $FLASK_PID
  exit 1
fi

echo "Túnel creado en: $NGROK_URL"

# 6. Registrar el webhook con Telegram
WEBHOOK_URL="$NGROK_URL/webhook"
echo "Registrando webhook de Telegram con URL: $WEBHOOK_URL"

curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook" -d "url=$WEBHOOK_URL"

echo "✅ Webhook registrado y bot en ejecución."

# Esperar a que termines con Ctrl+C
wait $FLASK_PID

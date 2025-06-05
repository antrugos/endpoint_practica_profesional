# Para su ejecución, debe seguir estos pasos
### 1. ejecutar python endpoint.py
### 2. Ejecutar en una terminal ngrok http 5000 (dejar corriendo)
### 3. Para la ejecución del Endpoint, inicia el codigo de python en una segunda terminal (dejar corriendo) y probar inicialmente en otra consola con esta url: curl -X POST "https://api.telegram.org/bot<token>/setWebhook?url=https://<url>.ngrok-free.app/webhook"
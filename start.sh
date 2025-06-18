#!/bin/bash

# Activar entorno virtual si lo tienes
# source venv/bin/activate

# Exportar variables de entorno si estás usando .env
export $(grep -v '^#' .env | xargs)

# Ejecutar FastAPI usando uvicorn
echo "Iniciando servidor FastAPI en http://0.0.0.0:8000"
uvicorn endpoint_fastAPI:app --host 0.0.0.0 --port 8000 --reload

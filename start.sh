#!/bin/bash
PORT=${PORT:-8080}
echo "Starting Gunicorn on port $PORT"
exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120 --max-requests 100 --max-requests-jitter 10 --worker-class gthread app:app


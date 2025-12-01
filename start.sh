#!/bin/bash
PORT=${PORT:-8080}
echo "Starting Gunicorn on port $PORT"
exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app


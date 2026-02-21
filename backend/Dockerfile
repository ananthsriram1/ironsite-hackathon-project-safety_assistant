FROM python:3.11

WORKDIR /app

COPY requirements.txt .
# ultralytics pulls in opencv-python (needs libGL); swap it for headless after install
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir "opencv-python-headless>=4.6.0"

# Source is bind-mounted at runtime for live reload (see docker-compose.yml)
# COPY . . is intentionally omitted here

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

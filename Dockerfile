FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY demo/ ./demo/
COPY embeddings/ ./embeddings/

EXPOSE 8501

CMD ["streamlit", "run", "demo/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
FROM python:3.12-slim
WORKDIR /app

# сначала зависимости — лучше кэшируется
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# копируем только то, что нужно для рантайма
COPY app.py ./app.py
COPY models/model.pkl ./model.pkl

ENV MODEL_PATH=/app/model.pkl
ENV PORT=8080
EXPOSE 8080

CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","8080"]

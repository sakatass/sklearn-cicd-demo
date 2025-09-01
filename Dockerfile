FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
# модель будет положена в образ CI-джобой (см. workflow ниже)
COPY app.py /app/app.py
COPY models/model.pkl /app/model.pkl
ENV MODEL_PATH=/app/model.pkl
ENV PORT=8080
EXPOSE 8080
CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","8080"]

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/app.py .
COPY models/credit_default_model.pkl ./models/

EXPOSE 5000

CMD ["python", "app.py"]

# Credit Default Prediction Service

## Описание
Веб-сервис для прогнозирования дефолта по кредитным картам. Модель обучена на датасете UCI "Default of Credit Card Clients".  
Проект демонстрирует полный цикл внедрения ML-модели: от обучения до контейнеризации и A/B-тестирования.

## Инструкция по запуску

### Локально (с виртуальным окружением)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/api/app.py

# Credit Default Prediction Service

## Описание проекта
Сервис прогнозирования дефолта по кредитным картам.  
Модель машинного обучения (Random Forest) обучена на датасете UCI «Default of Credit Card Clients».  
Цель — предсказать, допустит ли клиент дефолт в следующем месяце.

Проект демонстрирует полный цикл внедрения ML-модели в production: от обучения до контейнеризации, тестирования и A/B-тестирования.

## Быстрый старт

### Локально (Python 3.11+)
```bash
git clone https://github.com/chernova2601-pixel/ML_models_deploy_SC_Chernova_I.git
cd ML_models_deploy_SC_Chernova_I
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app/api.py
Docker
bash
docker build -f docker/Dockerfile -t credit_default_service .
docker run -d -p 5000:5000 --name credit_default credit_default_service
Docker Compose
bash
docker-compose up -d
Тестирование API
bash
python tests/test_api.py
API Endpoints
GET /health
Проверка работоспособности.
Ответ: {"status": "ok"}

POST /predict
Принимает JSON с 23 числовыми признаками.
Пример запроса:

bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1]*23}'
Ответ:

json
{"prediction": 0, "probability": 0.123}
Документация
Архитектура и обоснование
Проект реализован как монолитное приложение на Flask. Выбор обусловлен:

Небольшой размер и логическая целостность (один API, одна модель).

Отсутствие необходимости в независимом масштабировании компонентов.

Упрощение развёртывания и тестирования.

При гипотетическом масштабировании можно внедрить RabbitMQ для асинхронной обработки запросов, логирования и батч-предсказаний.

Логирование и мониторинг
Логи в формате JSON выводятся в stdout. В production-среде их можно собирать в ELK-стек (Elasticsearch, Logstash, Kibana) для централизованного мониторинга и анализа.

MLOps инструменты
DVC – контроль версий данных и моделей.

MLflow – управление экспериментами, логирование метрик и параметров.

Бизнес-метрики
Ожидаемые потери от дефолтов (Expected Loss) = PD * EAD * LGD. Снижение потерь за счёт лучшей калибровки модели.

Доля одобренных заявок при фиксированном уровне риска. Более точная модель позволяет одобрять больше клиентов без роста дефолтов.

A/B-тестирование
План описан в файле ab test plan.md.
Основные метрики: F1-score (класс дефолта) и Recall.
Трафик делится 50/50, продолжительность теста – 2 недели или до 10 000 событий.
Статистический тест – t-test, доверительные интервалы – бутстрап.

Структура репозитория
text
.
├── app/
│   ├── api.py               # Flask-приложение
│   └── model_handler.py     # Загрузка и инференс модели
├── models/
│   ├── model_v1.pkl         # Обученная модель
│   └── train_model.py       # Скрипт обучения
├── tests/
│   └── test_api.py          # Тесты API
├── docker/
│   └── Dockerfile           # Контейнеризация
├── notebooks/               (опционально)
├── requirements.txt
├── docker-compose.yml
├── ab test plan.md
└── README.md
Docker Hub
Образ доступен: docker pull chernova2601/credit_default_service:latest

Лицензия
MIT

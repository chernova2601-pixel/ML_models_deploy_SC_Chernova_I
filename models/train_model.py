"""
Скрипт обучения модели RandomForestClassifier на датасете UCI Credit Card.
Сохраняет модель в model_v1.pkl.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle
import os

def main():
    # Загрузка данных (попробуем локально или из URL)
    if os.path.exists('data/UCI_Credit_Card.csv'):
        df = pd.read_csv('data/UCI_Credit_Card.csv')
    else:
        # Скачиваем с UCI
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
        df = pd.read_excel(url, header=1)

    target = 'default.payment.next.month'
    X = df.drop(columns=[target, 'ID'])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=450,
        max_depth=16,
        min_samples_leaf=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f'F1-score on test: {f1:.4f}')

    with open('model_v1.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Model saved as model_v1.pkl')

if __name__ == '__main__':
    main()

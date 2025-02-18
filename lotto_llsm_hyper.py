import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os


# 데이터 로드 및 전처리 함수
def numbers2ohbin(numbers):
    ohbin = np.zeros(45)
    for i in range(6):
        ohbin[int(numbers[i]) - 1] = 1
    return ohbin


def ohbin2numbers(ohbin):
    numbers = []
    for i in range(len(ohbin)):
        if ohbin[i] == 1.0:
            numbers.append(i + 1)
    return numbers


# 데이터 준비
rows = np.loadtxt("./lotto.csv", delimiter=",")
row_count = len(rows)
numbers = rows[:, 1:7]
ohbins = list(map(numbers2ohbin, numbers))

x_samples = np.array(ohbins[0:row_count - 1])
y_samples = np.array(ohbins[1:row_count])

# 데이터 분할 및 형태 조정
x_train, x_test, y_train, y_test = train_test_split(x_samples, y_samples, test_size=0.2, random_state=42)
x_train = x_train.reshape(-1, 1, 45)
x_test = x_test.reshape(-1, 1, 45)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
x_val = x_val.reshape(-1, 1, 45)

# 하이퍼파라미터 설정
PARAMS = [
    {'units': 64, 'learning_rate': 0.001},
    {'units': 128, 'learning_rate': 0.001},
    {'units': 256, 'learning_rate': 0.001},
    {'units': 64, 'learning_rate': 0.0001},
    {'units': 128, 'learning_rate': 0.0001}
]


def create_model(units, learning_rate):
    inputs = keras.layers.Input(shape=(1, 45))
    x = keras.layers.LSTM(units=units)(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(45, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# 콜백 설정
def get_callbacks(model_index):
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]


# 모델 학습 및 평가 함수
def train_and_evaluate_model(params, model_index):
    model = create_model(params['units'], params['learning_rate'])

    history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks(model_index),
        verbose=1
    )

    return model, history, model.evaluate(x_test, y_test)


# 최적의 모델 찾기
best_accuracy = 0
best_model = None
best_history = None

print("Starting model training with different parameters...")

for i, params in enumerate(PARAMS):
    print(f"\nTraining model {i + 1}/{len(PARAMS)}")
    print(f"Parameters: {params}")

    try:
        model, history, (test_loss, test_acc) = train_and_evaluate_model(params, i)
        print(f"\nTest loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_history = history
            print("New best model found!")
    except Exception as e:
        print(f"Error training model with parameters {params}: {e}")
        continue

if best_model is not None:
    print("\nTraining completed!")
    print(f"Best test accuracy: {best_accuracy:.4f}")


    # 로또 번호 생성 함수
    def gen_numbers_from_probability(prob_dist):
        prob_dist = np.clip(prob_dist, 0, 1)  # 확률값을 0과 1 사이로 제한
        prob_dist = prob_dist / prob_dist.sum()  # 정규화
        numbers = np.random.choice(
            range(1, 46),
            size=6,
            replace=False,
            p=prob_dist
        )
        return sorted(numbers)


    # 예측 실행
    print('\nPredicted numbers:')
    xs = x_samples[-1].reshape(1, 1, 45)
    ys_pred = best_model.predict(xs, verbose=0)

    for n in range(5):
        numbers = gen_numbers_from_probability(ys_pred[0])
        print(f'Set {n + 1}: {numbers}')

    # 모델 성능 통계 출력
    print("\nModel Performance Statistics:")
    val_accuracy = max(best_history.history['val_accuracy'])
    print(f"Best Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Test Accuracy: {best_accuracy:.4f}")

else:
    print("No successful model training. Please check the error messages above.")
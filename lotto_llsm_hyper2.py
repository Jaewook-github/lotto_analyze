import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
x_train, x_temp, y_train, y_temp = train_test_split(x_samples, y_samples, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 데이터 reshape
x_train = x_train.reshape(-1, 1, 45)
x_val = x_val.reshape(-1, 1, 45)
x_test = x_test.reshape(-1, 1, 45)


def create_model():
    inputs = keras.layers.Input(shape=(1, 45))

    # LSTM layers with residual connections
    x = keras.layers.LSTM(256, return_sequences=True)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.LSTM(128)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    # Dense layers
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(45, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile with optimized parameters
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# 콜백 정의
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
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

# 모델 생성 및 학습
print("Starting model training...")
model = create_model()

history = model.fit(
    x_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# 학습 과정 시각화
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# 로또 번호 생성 함수
def gen_numbers_from_probability(prob_dist):
    prob_dist = np.clip(prob_dist, 0, 1)  # Clip probabilities between 0 and 1
    prob_dist = prob_dist / np.sum(prob_dist)  # Normalize probabilities
    numbers = np.random.choice(
        range(1, 46),
        size=6,
        replace=False,
        p=prob_dist
    )
    return sorted(numbers)


# 예측 및 결과 출력
print('\nPredicted Lotto Numbers:')
xs = x_samples[-1].reshape(1, 1, 45)
ys_pred = model.predict(xs, verbose=0)

for n in range(5):
    numbers = gen_numbers_from_probability(ys_pred[0])
    print(f'Set {n + 1}: {numbers}')

# 모델 성능 요약
print("\nModel Performance Summary:")
val_accuracy = max(history.history['val_accuracy'])
print(f"Best Validation Accuracy: {val_accuracy:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Best Training Accuracy: {max(history.history['accuracy']):.4f}")
print(f"Lowest Validation Loss: {min(history.history['val_loss']):.4f}")
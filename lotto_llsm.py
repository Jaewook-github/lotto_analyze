import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 데이터 로드
rows = np.loadtxt("./lotto.csv", delimiter=",")
row_count = len(rows)
print(f"Total rows: {row_count}")

# 당첨번호를 원핫인코딩벡터로 변환
def numbers2ohbin(numbers):
    ohbin = np.zeros(45)
    for i in range(6):
        ohbin[int(numbers[i]) - 1] = 1
    return ohbin

# 원핫인코딩벡터를 번호로 변환
def ohbin2numbers(ohbin):
    return [i + 1 for i in range(len(ohbin)) if ohbin[i] == 1.0]

numbers = rows[:, 1:7]
ohbins = np.array([numbers2ohbin(num) for num in numbers])

x_samples = ohbins[:-1]
y_samples = ohbins[1:]

print("Sample input:", ohbin2numbers(x_samples[0]))
print("Sample output:", ohbin2numbers(y_samples[0]))

# 데이터 분할
train_idx = (0, 1000)
val_idx = (1000, 1100)
test_idx = (1100, len(x_samples))

print(f"Train: {train_idx}, Val: {val_idx}, Test: {test_idx}")

# 데이터셋 생성
def create_dataset(x, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(x_samples[train_idx[0]:train_idx[1]], y_samples[train_idx[0]:train_idx[1]], 32)
val_dataset = create_dataset(x_samples[val_idx[0]:val_idx[1]], y_samples[val_idx[0]:val_idx[1]], 32)
test_dataset = create_dataset(x_samples[test_idx[0]:test_idx[1]], y_samples[test_idx[0]:test_idx[1]], 32)

# 모델 정의
inputs = keras.layers.Input(shape=(45,))
expanded_inputs = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)  # TensorFlow 함수를 Lambda 레이어로 감싸기
lstm = keras.layers.LSTM(128, return_sequences=False)(expanded_inputs)
outputs = keras.layers.Dense(45, activation='sigmoid')(lstm)
model = keras.Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# 콜백 정의
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# 모델 훈련
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[early_stopping],
    verbose=1
)

# 결과 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()

# 88회부터 지금까지 1등부터 5등까지 상금의 평균
mean_prize = [np.mean(rows[87:, 8 + i]) for i in range(5)]
print("Mean prizes:", mean_prize)

# 등수와 상금을 반환
def calc_reward(true_numbers, true_bonus, pred_numbers):
    count = sum([1 for ps in pred_numbers if ps in true_numbers])

    if count == 6:
        return 0, mean_prize[0]
    elif count == 5 and true_bonus in pred_numbers:
        return 1, mean_prize[1]
    elif count == 5:
        return 2, mean_prize[2]
    elif count == 4:
        return 3, mean_prize[3]
    elif count == 3:
        return 4, mean_prize[4]

    return 5, 0

def gen_numbers_from_probability(nums_prob):
    ball_box = []
    for n in range(45):
        ball_count = int(nums_prob[n] * 100 + 1)
        ball_box += [n + 1] * ball_count
    return sorted(np.random.choice(ball_box, 6, replace=False))

# 모델 평가
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

total_reward = np.zeros(3)
total_grade = np.zeros((3, 6), dtype=int)

for i, dataset in enumerate([train_dataset, val_dataset, test_dataset]):
    for x, y in dataset:
        pred = model.predict(x)
        for _ in range(10):  # 10번의 로또 구매 시뮬레이션
            numbers = gen_numbers_from_probability(pred[0])
            grade, reward = calc_reward(y[0], rows[i + 1, 7], numbers)
            total_reward[i] += reward
            total_grade[i, grade] += 1

print("\nResults:")
print(f"Train: {total_grade[0]} - Reward: {total_reward[0]:,}")
print(f"Val  : {total_grade[1]} - Reward: {total_reward[1]:,}")
print(f"Test : {total_grade[2]} - Reward: {total_reward[2]:,}")

# 다음 회차 예측
last_numbers = x_samples[-1].reshape(1, 45)
next_pred = model.predict(last_numbers)

print("\nPredicted numbers for next draw:")
for _ in range(5):
    numbers = gen_numbers_from_probability(next_pred[0])
    print(numbers)

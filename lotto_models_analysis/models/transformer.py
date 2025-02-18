
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LayerNormalization,
                                     MultiHeadAttention, Dropout, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .base import BaseModel
import logging


class TransformerBlock(tf.keras.layers.Layer):
    """트랜스포머 블록 구현"""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        # 멀티헤드 어텐션
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # 피드포워드 네트워크
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(BaseModel):
    """트랜스포머 기반 로또 번호 예측 모델"""

    def __init__(self):
        super().__init__()
        # 모델 하이퍼파라미터
        self.embedding_dim = 64
        self.num_heads = 8
        self.ff_dim = 64
        self.dropout_rate = 0.1
        self.learning_rate = 0.001
        self.sequence_length = 10

        self.build_model()

    def build_model(self):
        """트랜스포머 모델 구축"""
        try:
            # 입력층
            inputs = Input(shape=(self.sequence_length, 45))

            # 위치 인코딩 추가
            pos_encoding = self.positional_encoding(
                self.sequence_length,
                self.embedding_dim
            )

            # 입력 임베딩
            x = Dense(self.embedding_dim)(inputs)
            x = x + pos_encoding

            # 트랜스포머 블록들
            for _ in range(4):  # 4개의 트랜스포머 블록 스택
                x = TransformerBlock(
                    self.embedding_dim,
                    self.num_heads,
                    self.ff_dim,
                    self.dropout_rate
                )(x)

            # 출력층
            x = GlobalAveragePooling1D()(x)
            x = Dropout(0.1)(x)
            x = Dense(256, activation="relu")(x)
            x = Dropout(0.1)(x)
            outputs = Dense(45, activation="softmax")(x)

            # 모델 생성
            self.model = Model(inputs=inputs, outputs=outputs)

            # 모델 컴파일
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            logging.info("트랜스포머 모델 구축 완료")
            self.model.summary(print_fn=logging.info)

        except Exception as e:
            logging.error(f"모델 구축 중 오류 발생: {str(e)}")
            raise

    def positional_encoding(self, position, d_model):
        """위치 인코딩 생성"""
        try:
            angle_rads = self.get_angles(
                np.arange(position)[:, np.newaxis],
                np.arange(d_model)[np.newaxis, :],
                d_model
            )

            # 짝수 위치에는 sin 적용
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

            # 홀수 위치에는 cos 적용
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

            pos_encoding = angle_rads[np.newaxis, ...]

            return tf.cast(pos_encoding, dtype=tf.float32)

        except Exception as e:
            logging.error(f"위치 인코딩 생성 중 오류: {str(e)}")
            raise

    def get_angles(self, pos, i, d_model):
        """위치 인코딩을 위한 각도 계산"""
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def prepare_data(self, historical_data):
        """데이터 전처리"""
        try:
            X = []
            y = []

            # 시퀀스 데이터 생성
            for i in range(len(historical_data) - self.sequence_length):
                sequence = historical_data[i:i + self.sequence_length]
                target = historical_data[i + self.sequence_length]

                # 원-핫 인코딩
                sequence_encoded = np.zeros((self.sequence_length, 45))
                target_encoded = np.zeros(45)

                for j, numbers in enumerate(sequence):
                    sequence_encoded[j, np.array(numbers) - 1] = 1
                target_encoded[np.array(target) - 1] = 1

                X.append(sequence_encoded)
                y.append(target_encoded)

            return np.array(X), np.array(y)

        except Exception as e:
            logging.error(f"데이터 전처리 중 오류: {str(e)}")
            raise

    def train(self, train_data, validation_data=None):
        """모델 학습"""
        try:
            X, y = self.prepare_data(train_data)

            # 검증 데이터 준비
            if validation_data is not None:
                X_val, y_val = self.prepare_data(validation_data)
                validation_data = (X_val, y_val)

            # 콜백 설정
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ),
                tf.keras.callbacks.ModelCheckpoint(  # 오타 수정
                    filepath='best_models/transformer_model.keras',
                    monitor='val_loss' if validation_data else 'loss',
                    save_best_only=True
                )
            ]

            # 모델 학습
            history = self.model.fit(
                X, y,
                epochs=100,
                batch_size=32,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )

            # 학습 결과 로깅
            self.log_training_results(history)

            return history

        except Exception as e:
            logging.error(f"모델 학습 중 오류: {str(e)}")
            raise

    # def predict_numbers(self, input_data=None, n_predictions=6):
    #     """번호 예측"""
    #     try:
    #         if input_data is None:
    #             raise ValueError("입력 데이터가 필요합니다.")
    #
    #         # 입력 데이터 준비
    #         sequence = input_data[-self.sequence_length:]
    #         sequence_encoded = np.zeros((1, self.sequence_length, 45))
    #
    #         for i, numbers in enumerate(sequence):
    #             sequence_encoded[0, i, np.array(numbers) - 1] = 1
    #
    #         # 예측 수행
    #         predictions = self.model.predict(sequence_encoded, verbose=0)[0]
    #
    #         # 번호 선택
    #         selected_numbers = []
    #         remaining_probs = predictions.copy()
    #
    #         for _ in range(n_predictions):
    #             # 이미 선택된 번호 제외
    #             remaining_probs[np.array(selected_numbers) - 1] = 0
    #
    #             # 확률 정규화
    #             remaining_probs = remaining_probs / np.sum(remaining_probs)
    #
    #             # 번호 선택
    #             selected = np.random.choice(45, p=remaining_probs) + 1
    #             selected_numbers.append(selected)
    #
    #         result = self.format_numbers(selected_numbers)
    #
    #         # 예측 결과 로깅
    #         prob_str = ", ".join(f"{predictions[n - 1]:.4f}" for n in result)
    #         logging.info(f"예측된 번호: {result}")
    #         logging.info(f"각 번호의 확률: [{prob_str}]")
    #
    #         return result
    #
    #     except Exception as e:
    #         logging.error(f"예측 중 오류: {str(e)}")
    #         raise

    def predict_numbers(self, input_data=None, n_predictions=6):
        """번호 예측"""
        try:
            if input_data is None:
                raise ValueError("입력 데이터가 필요합니다.")

            # 입력 데이터 준비
            sequence = input_data[-self.sequence_length:]
            sequence_encoded = np.zeros((1, self.sequence_length, 45))

            for i, nums in enumerate(sequence):
                for num in nums:
                    sequence_encoded[0, i, int(num - 1)] = 1

            # 예측 수행
            predictions = self.model.predict(sequence_encoded, verbose=0)[0]

            # 번호 선택
            selected_numbers = []
            remaining_probs = predictions.copy()

            for _ in range(n_predictions):
                # 이미 선택된 번호 제외
                mask = np.ones(45, dtype=bool)
                if selected_numbers:
                    mask[np.array([int(x - 1) for x in selected_numbers])] = False

                # 확률 정규화
                masked_probs = remaining_probs * mask
                if np.sum(masked_probs) > 0:
                    masked_probs = masked_probs / np.sum(masked_probs)
                    selected = int(np.random.choice(45, p=masked_probs)) + 1
                else:
                    # 확률이 0인 경우 무작위 선택
                    available = np.where(mask)[0]
                    selected = int(np.random.choice(available)) + 1

                selected_numbers.append(selected)

            result = self.format_numbers(selected_numbers)

            # 예측 결과 로깅
            prob_str = ", ".join(f"{predictions[n - 1]:.4f}" for n in result)
            logging.info(f"예측된 번호: {result}")
            logging.info(f"각 번호의 확률: [{prob_str}]")

            return result

        except Exception as e:
            logging.error(f"예측 중 오류: {str(e)}")
            raise

    def log_training_results(self, history):
        """학습 결과 로깅"""
        logging.info("\n=== 학습 결과 ===")
        logging.info(f"최종 훈련 손실: {history.history['loss'][-1]:.4f}")
        logging.info(f"최종 훈련 정확도: {history.history['accuracy'][-1]:.4f}")

        if 'val_loss' in history.history:
            logging.info(f"최종 검증 손실: {history.history['val_loss'][-1]:.4f}")
            logging.info(f"최종 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")

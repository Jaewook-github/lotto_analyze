
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, LSTM, Conv2D, MaxPooling2D, Flatten,
                                     Dropout, Input, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from .base import BaseModel
import logging


class HybridModel(BaseModel):
    """CNN과 LSTM을 결합한 하이브리드 모델"""

    def __init__(self):
        super().__init__()
        self.embedding_dim = 64
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        self.build_model()

    def build_model(self):
        """CNN과 LSTM을 결합한 모델 구축"""
        try:
            # CNN 입력 브랜치
            cnn_input = Input(shape=(7, 7, 1), name='marking_pattern_input')
            x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(cnn_input)
            x = BatchNormalization(name='batch_conv1')(x)
            x = MaxPooling2D((2, 2), name='pool1')(x)
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
            x = BatchNormalization(name='batch_conv2')(x)
            x = MaxPooling2D((2, 2), name='pool2')(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
            x = BatchNormalization(name='batch_conv3')(x)
            x = Flatten(name='flatten')(x)
            x = Dense(256, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                      name='cnn_dense1')(x)
            x = BatchNormalization(name='batch_cnn_dense1')(x)
            cnn_output = Dense(128, activation='relu', name='cnn_dense2')(x)

            # LSTM 입력 브랜치
            lstm_input = Input(shape=(None, 45), name='sequence_input')
            y = LSTM(128, return_sequences=True, name='lstm1')(lstm_input)
            y = BatchNormalization(name='batch_lstm1')(y)
            y = LSTM(64, return_sequences=True, name='lstm2')(y)
            y = BatchNormalization(name='batch_lstm2')(y)
            y = LSTM(32, name='lstm3')(y)
            y = BatchNormalization(name='batch_lstm3')(y)
            lstm_output = Dense(128, activation='relu', name='lstm_dense')(y)

            # 모델 결합
            combined = tf.keras.layers.concatenate([cnn_output, lstm_output],
                                                   name='combine_branches')

            # 최종 출력층
            z = Dense(512, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                      name='combined_dense1')(combined)
            z = BatchNormalization(name='batch_combined1')(z)
            z = Dropout(self.dropout_rate, name='dropout1')(z)
            z = Dense(256, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                      name='combined_dense2')(z)
            z = BatchNormalization(name='batch_combined2')(z)
            z = Dropout(self.dropout_rate * 0.8, name='dropout2')(z)
            z = Dense(128, activation='relu', name='combined_dense3')(z)
            z = BatchNormalization(name='batch_combined3')(z)
            output = Dense(45, activation='softmax', name='number_predictions')(z)

            # 모델 생성
            self.model = Model(inputs=[cnn_input, lstm_input], outputs=output)

            # 옵티마이저 설정
            optimizer = Adam(learning_rate=self.learning_rate)

            # 모델 컴파일
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            logging.info("하이브리드 모델 구축 완료")

        except Exception as e:
            logging.error(f"모델 구축 중 오류 발생: {str(e)}")
            raise

    def train(self, train_data, validation_data=None):
        """모델 학습"""
        try:
            X_cnn_train, X_lstm_train, y_train = train_data

            # 데이터 크기 확인 및 로깅
            logging.info(f"Training data shapes:")
            logging.info(f"X_cnn: {X_cnn_train.shape}")
            logging.info(f"X_lstm: {X_lstm_train.shape}")
            logging.info(f"y: {y_train.shape}")


            if validation_data:
                X_cnn_val, X_lstm_val, y_val = validation_data
                val_data = ([X_cnn_val, X_lstm_val], y_val)
            else:
                val_data = None

            # 콜백 설정
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='best_models/hybrid_model.keras',
                    monitor='val_accuracy' if validation_data else 'accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]

            # 모델 학습
            history = self.model.fit(
                [X_cnn_train, X_lstm_train],
                y_train,
                epochs=100,
                batch_size=32,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )

            # 학습 결과 로깅
            self.log_training_results(history)

            return history

        except Exception as e:
            logging.error(f"모델 학습 중 오류 발생: {str(e)}")
            raise

    # def predict_numbers(self, input_data=None, n_predictions=6):
    #     """번호 예측"""
    #     try:
    #         X_cnn, X_lstm = input_data
    #         predictions = self.model.predict([X_cnn[-1:], X_lstm[-1:]], verbose=0)
    #
    #         # 예측 확률 검증
    #         if np.any(np.isnan(predictions)):
    #             raise ValueError("예측값에 NaN이 포함되어 있습니다.")
    #
    #         # 확률 정규화
    #         probs = predictions[0]
    #         probs = np.clip(probs, 1e-10, 1.0)
    #         probs = probs / np.sum(probs)
    #
    #         # 번호 선택
    #         selected_numbers = []
    #         remaining_probs = probs.copy()
    #
    #         for _ in range(n_predictions):
    #             # 이미 선택된 번호 제외
    #             mask = np.ones_like(remaining_probs, dtype=bool)
    #             mask[np.array(selected_numbers) - 1] = False
    #
    #             # 남은 번호들의 확률 재조정
    #             masked_probs = remaining_probs * mask
    #             if masked_probs.sum() > 0:
    #                 masked_probs = masked_probs / masked_probs.sum()
    #                 selected = np.random.choice(45, p=masked_probs) + 1
    #             else:
    #                 # 확률이 0인 경우 무작위 선택
    #                 available = np.where(mask)[0]
    #                 selected = np.random.choice(available) + 1
    #
    #             selected_numbers.append(selected)
    #
    #         # 결과 검증 및 정렬
    #         result = self.format_numbers(selected_numbers)
    #
    #         # 예측 결과 로깅
    #         self.log_prediction_results(result, probs)
    #
    #         return result
    #
    #     except Exception as e:
    #         logging.error(f"번호 예측 중 오류 발생: {str(e)}")
    #         raise

    def predict_numbers(self, input_data=None, n_predictions=6):
        """번호 예측"""
        try:
            X_cnn, X_lstm = input_data
            predictions = self.model.predict([X_cnn[-1:], X_lstm[-1:]], verbose=0)

            # 예측 확률 검증
            if np.any(np.isnan(predictions)):
                raise ValueError("예측값에 NaN이 포함되어 있습니다.")

            # 확률 정규화
            probs = predictions[0]
            probs = np.clip(probs, 1e-10, 1.0)
            probs = probs / np.sum(probs)

            # 번호 선택
            selected_numbers = []
            remaining_probs = probs.copy()

            while len(selected_numbers) < n_predictions:
                # 이미 선택된 번호 제외
                mask = np.ones(45, dtype=bool)
                if selected_numbers:
                    mask[np.array([int(x - 1) for x in selected_numbers])] = False

                # 남은 번호들의 확률 재조정
                masked_probs = remaining_probs * mask
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
                    selected = int(np.random.choice(45, p=masked_probs)) + 1
                else:
                    # 확률이 0인 경우 무작위 선택
                    available = np.where(mask)[0]
                    selected = int(np.random.choice(available)) + 1

                selected_numbers.append(selected)

            # 결과 검증 및 정렬
            result = self.format_numbers(selected_numbers)

            # 예측 결과 로깅
            self.log_prediction_results(result, probs)

            return result

        except Exception as e:
            logging.error(f"번호 예측 중 오류 발생: {str(e)}")
            raise

    def log_training_results(self, history):
        """학습 결과 로깅"""
        logging.info("\n=== 학습 결과 ===")
        logging.info(f"최종 훈련 손실: {history.history['loss'][-1]:.4f}")
        logging.info(f"최종 훈련 정확도: {history.history['accuracy'][-1]:.4f}")

        if 'val_loss' in history.history:
            logging.info(f"최종 검증 손실: {history.history['val_loss'][-1]:.4f}")
            logging.info(f"최종 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")

    def log_prediction_results(self, numbers, probabilities):
        """예측 결과 로깅"""
        logging.info("\n=== 예측 결과 ===")
        logging.info(f"선택된 번호: {numbers}")
        prob_str = ", ".join(f"{probabilities[n - 1]:.4f}" for n in numbers)
        logging.info(f"각 번호의 확률: [{prob_str}]")

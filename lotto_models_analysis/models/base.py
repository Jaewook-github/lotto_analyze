
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
import logging

class BaseModel(ABC):
    """모든 예측 모델의 기본 클래스"""

    def __init__(self):
        self.model = None
        self.model_name = self.__class__.__name__

    @abstractmethod
    def build_model(self):
        """모델 구조 생성"""
        pass

    @abstractmethod
    def train(self, train_data, validation_data=None):
        """모델 학습"""
        pass

    @abstractmethod
    def predict_numbers(self, input_data=None, n_predictions=6):
        """번호 예측"""
        pass

    def save_model(self, filepath):
        """모델 저장"""
        if self.model is not None:
            try:
                if isinstance(self.model, tf.keras.Model):
                    self.model.save(f"{filepath}.keras")
                else:
                    np.save(f"{filepath}.npy", self.model)
            except Exception as e:
                print(f"모델 저장 중 오류 발생: {str(e)}")

    def load_best_model(self):
        """최고 성능 모델 로드"""
        try:
            model_path = f'best_models/{self.__class__.__name__.lower()}_model'
            if os.path.exists(f"{model_path}.keras"):
                self.load_model(f"{model_path}.keras")
            elif os.path.exists(f"{model_path}.npy"):
                self.load_model(f"{model_path}.npy")
            else:
                logging.warning(f"저장된 모델을 찾을 수 없습니다: {model_path}")
        except Exception as e:
            logging.error(f"모델 로드 중 오류: {str(e)}")


    def load_model(self, filepath):
        """모델 로드"""
        try:
            if filepath.endswith('.keras'):
                self.model = tf.keras.models.load_model(filepath)
            elif filepath.endswith('.npy'):
                self.model = np.load(filepath, allow_pickle=True)
            else:
                raise ValueError("지원하지 않는 파일 형식입니다.")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")

    # def validate_numbers(self, numbers):
    #     """예측된 번호의 유효성 검사"""
    #     if not isinstance(numbers, (list, np.ndarray)):
    #         return False
    #
    #     if len(numbers) != 6:
    #         return False
    #
    #     numbers = np.array(numbers)
    #     if not np.all((numbers >= 1) & (numbers <= 45)):
    #         return False
    #
    #     if len(np.unique(numbers)) != 6:
    #         return False
    #
    #     return True

    def validate_numbers(self, numbers):
        """예측된 번호의 유효성 검사"""
        try:
            if not isinstance(numbers, (list, np.ndarray)):
                return False

            # numpy array를 list로 변환
            numbers = list(map(int, numbers))

            if len(numbers) != 6:
                return False

            # 각 번호가 1-45 사이인지 확인
            if not all(1 <= num <= 45 for num in numbers):
                return False

            # 중복 번호가 없는지 확인
            if len(set(numbers)) != 6:
                return False

            return True

        except Exception as e:
            logging.error(f"번호 검증 중 오류: {str(e)}")
            return False

    def format_numbers(self, numbers):
        """번호 형식 통일화"""
        if not self.validate_numbers(numbers):
            raise ValueError("유효하지 않은 번호 형식입니다.")

        return sorted(list(map(int, numbers)))

    def evaluate_prediction(self, predicted, actual):
        """예측 결과 평가"""
        if not (self.validate_numbers(predicted) and
                self.validate_numbers(actual)):
            raise ValueError("유효하지 않은 번호입니다.")

        matches = len(set(predicted) & set(actual))

        # 등수 결정
        if matches == 6:
            return 1  # 1등
        elif matches == 5:
            return 3  # 3등 (보너스 번호 제외)
        elif matches == 4:
            return 4  # 4등
        elif matches == 3:
            return 5  # 5등
        else:
            return 0  # 낙첨

    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'model_type': str(type(self.model)) if self.model else None,
            'trainable_params': self.get_trainable_params() if self.model else 0
        }

    def get_trainable_params(self):
        """학습 가능한 파라미터 수 계산"""
        if isinstance(self.model, tf.keras.Model):
            return self.model.count_params()
        return 0

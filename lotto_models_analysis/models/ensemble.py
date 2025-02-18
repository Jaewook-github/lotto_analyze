import numpy as np
from .base import BaseModel
from .hybrid import HybridModel
from .reinforcement import ReinforcementLearningModel
from .genetic import GeneticAlgorithmModel
from .transformer import TransformerModel
from .statistical import StatisticalModel
import logging
from datetime import datetime


class EnsembleModel(BaseModel):
    """여러 모델을 결합한 앙상블 모델"""

    def __init__(self):
        super().__init__()
        # 기본 가중치 설정
        self.weights = {
            "혼합 모델": 0.3,
            "강화학습 모델": 0.2,
            "유전 알고리즘 모델": 0.2,
            "트랜스포머 모델": 0.15,
            "통계 기반 모델": 0.15
        }

        # 각 모델 초기화
        self.models = {
            "혼합 모델": HybridModel(),
            "강화학습 모델": ReinforcementLearningModel(),
            "유전 알고리즘 모델": GeneticAlgorithmModel(),
            "트랜스포머 모델": TransformerModel(),
            "통계 기반 모델": StatisticalModel()
        }

        # 성능 기록
        self.model_performance = {name: [] for name in self.models.keys()}
        self.setup_logging()

    def build_model(self):
        """모델 구조 생성 - 앙상블 모델은 별도의 모델 구조가 필요 없음"""
        # 각 하위 모델들의 build_model 호출
        for model in self.models.values():
            model.build_model()

    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            filename=f'logs/ensemble_{datetime.now():%Y%m%d_%H%M%S}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )

    def train_all_models(self, train_data, validation_data=None):
        """모든 모델 학습"""
        try:
            for name, model in self.models.items():
                logging.info(f"\n=== {name} 학습 시작 ===")
                try:
                    model.train(train_data, validation_data)

                    # 검증 데이터로 성능 평가
                    if validation_data is not None:
                        performance = self.evaluate_model(model, validation_data)
                        self.model_performance[name].append(performance)
                        logging.info(f"{name} 성능: {performance:.4f}")

                    np.save('best_models/ensemble_weight.npy', self.weights)

                except Exception as e:
                    logging.error(f"{name} 학습 중 오류: {str(e)}")

            # 성능에 따른 가중치 업데이트
            self.update_weights()

        except Exception as e:
            logging.error(f"앙상블 모델 학습 중 오류: {str(e)}")
            raise

    def train(self, train_data, validation_data=None):
        """학습 인터페이스 구현"""
        return self.train_all_models(train_data, validation_data)

    def evaluate_model(self, model, validation_data):
        """개별 모델 성능 평가"""
        try:
            correct_predictions = 0
            total_predictions = len(validation_data)

            for actual_numbers in validation_data:
                predicted = model.predict_numbers()
                matches = len(set(predicted) & set(actual_numbers))

                # 매칭 개수에 따른 점수 부여
                if matches == 6:
                    correct_predictions += 1.0
                elif matches == 5:
                    correct_predictions += 0.5
                elif matches == 4:
                    correct_predictions += 0.2
                elif matches == 3:
                    correct_predictions += 0.1

            return correct_predictions / total_predictions

        except Exception as e:
            logging.error(f"모델 평가 중 오류: {str(e)}")
            raise

    def update_weights(self):
        """모델 성능에 따른 가중치 업데이트"""
        try:
            total_performance = 0
            new_weights = {}

            # 각 모델의 평균 성능 계산
            for name in self.models.keys():
                if self.model_performance[name]:
                    avg_performance = np.mean(self.model_performance[name])
                    total_performance += avg_performance
                    new_weights[name] = avg_performance

            # 가중치 정규화
            if total_performance > 0:
                for name in new_weights:
                    new_weights[name] /= total_performance
                self.weights = new_weights

            logging.info("\n=== 업데이트된 가중치 ===")
            for name, weight in self.weights.items():
                logging.info(f"{name}: {weight:.4f}")

        except Exception as e:
            logging.error(f"가중치 업데이트 중 오류: {str(e)}")
            raise

    # def predict_numbers(self, input_data=None, n_predictions=6):
    #     """앙상블 예측 수행"""
    #     try:
    #         all_predictions = {}
    #         final_probabilities = np.zeros(45)
    #
    #         # 각 모델의 예측 수행
    #         for name, model in self.models.items():
    #             try:
    #                 model_predictions = model.predict_numbers(
    #                     input_data=input_data,
    #                     n_predictions=n_predictions
    #                 )
    #                 all_predictions[name] = model_predictions
    #
    #                 # 예측을 확률 분포로 변환
    #                 prob_dist = np.zeros(45)
    #                 for num in model_predictions:
    #                     prob_dist[num - 1] += 1
    #                 prob_dist = prob_dist / np.sum(prob_dist)
    #
    #                 # 가중치 적용
    #                 final_probabilities += prob_dist * self.weights.get(name, 0)
    #
    #             except Exception as e:
    #                 logging.error(f"{name} 예측 중 오류: {str(e)}")
    #
    #         # 최종 예측
    #         selected_numbers = []
    #         remaining_probs = final_probabilities.copy()
    #
    #         while len(selected_numbers) < n_predictions:
    #             # 이미 선택된 번호 제외
    #             mask = np.ones_like(remaining_probs, dtype=bool)
    #             mask[np.array(selected_numbers) - 1] = False
    #
    #             # 확률 정규화
    #             masked_probs = remaining_probs * mask
    #             if masked_probs.sum() > 0:
    #                 masked_probs = masked_probs / masked_probs.sum()
    #                 selected = np.random.choice(45, p=masked_probs) + 1
    #                 selected_numbers.append(selected)
    #
    #         result = self.format_numbers(selected_numbers)
    #
    #         # 예측 결과 로깅
    #         self.log_prediction_results(result, all_predictions)
    #
    #         return result
    #
    #     except Exception as e:
    #         logging.error(f"앙상블 예측 중 오류: {str(e)}")
    #         raise

    def predict_numbers(self, input_data=None, n_predictions=6):
        """앙상블 예측 수행"""
        try:
            all_predictions = []
            final_probabilities = np.zeros(45)

            # 각 모델의 예측 수행
            for name, model in self.models.items():
                try:
                    # 각 모델별 예측 수행
                    if name == "혼합 모델":
                        pred = model.predict_numbers(input_data)
                    elif name == "강화학습 모델":
                        state = np.zeros(45)  # 초기 상태
                        pred = model.predict_numbers(state.reshape(1, -1))
                    else:
                        pred = model.predict_numbers(input_data)

                    if pred is not None:
                        all_predictions.append(pred)

                        # 예측을 확률 분포로 변환
                        prob_dist = np.zeros(45)
                        for num in pred:
                            prob_dist[int(num - 1)] = 1
                        prob_dist = prob_dist / np.sum(prob_dist)

                        # 가중치 적용
                        final_probabilities += prob_dist * self.weights.get(name, 0)

                except Exception as e:
                    logging.error(f"{name} 예측 중 오류: {str(e)}")
                    continue

            # 충분한 예측이 없는 경우
            if not all_predictions:
                raise ValueError("유효한 예측 결과가 없습니다.")

            # 최종 예측
            final_numbers = []
            remaining_probs = final_probabilities.copy()

            for _ in range(n_predictions):
                if len(final_numbers) >= n_predictions:
                    break

                # 이미 선택된 번호 마스킹
                mask = np.ones(45, dtype=bool)
                selected_indices = [int(x - 1) for x in final_numbers]
                mask[selected_indices] = False

                # 남은 확률 계산
                masked_probs = remaining_probs * mask
                if np.sum(masked_probs) > 0:
                    masked_probs = masked_probs / np.sum(masked_probs)
                    selected = int(np.argmax(masked_probs)) + 1
                    final_numbers.append(selected)
                else:
                    # 확률이 0인 경우 무작위 선택
                    available = list(set(range(1, 46)) - set(final_numbers))
                    selected = int(np.random.choice(available))
                    final_numbers.append(selected)

            result = sorted(final_numbers)
            logging.info(f"앙상블 최종 예측: {result}")
            return result

        except Exception as e:
            logging.error(f"앙상블 예측 중 오류: {str(e)}")
            raise


    def log_prediction_results(self, final_numbers, model_predictions):
        """예측 결과 로깅"""
        logging.info("\n=== 앙상블 예측 결과 ===")
        logging.info(f"최종 예측 번호: {final_numbers}")

        logging.info("\n각 모델별 예측:")
        for name, predictions in model_predictions.items():
            logging.info(f"{name}: {predictions}")

        logging.info("\n현재 모델 가중치:")
        for name, weight in self.weights.items():
            logging.info(f"{name}: {weight:.4f}")

    def get_model_insights(self):
        """모델 분석 정보 반환"""
        return {
            'weights': self.weights,
            'performance': {
                name: np.mean(perfs) if perfs else 0
                for name, perfs in self.model_performance.items()
            },
            'predictions_by_model': {
                name: model.get_model_info()
                for name, model in self.models.items()
            }
        }

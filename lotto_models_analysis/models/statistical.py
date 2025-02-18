
import numpy as np
from collections import defaultdict
from .base import BaseModel
import logging
import pandas as pd


class StatisticalModel(BaseModel):
    """통계 기반 로또 번호 예측 모델"""

    # def __init__(self):
    #     super().__init__()
    #     # 통계 데이터 저장소
    #     self.number_freq = np.zeros(45)  # 번호별 출현 빈도
    #     self.pair_freq = np.zeros((45, 45))  # 번호 쌍 출현 빈도
    #     self.gap_freq = defaultdict(int)  # 번호 간격 빈도
    #     self.section_freq = np.zeros(5)  # 구간별 출현 빈도
    #     self.streak_data = defaultdict(int)  # 연속 출현/미출현 데이터
    #     self.recent_numbers = []  # 최근 당첨 번호
    #     self.number_stats = {}  # 상세 통계 정보

    def __init__(self):
        super().__init__()
        # 통계 데이터 저장소
        self.number_freq = np.zeros(45)  # 번호별 출현 빈도
        self.pair_freq = np.zeros((45, 45))  # 번호 쌍 출현 빈도
        self.gap_freq = defaultdict(int)  # 번호 간격 빈도
        self.section_freq = np.zeros(5)  # 구간별 출현 빈도
        self.streak_data = defaultdict(int)  # 연속 출현/미출현 데이터
        self.recent_numbers = []  # 최근 당첨 번호
        self.number_stats = {}  # 상세 통계 정보

    def build_model(self):
        """모델 구조 생성 - 통계 모델은 별도의 딥러닝 모델 구조가 필요 없음"""
        pass

    def analyze_data(self, historical_data):
        """데이터 분석"""
        try:
            logging.info("통계 분석 시작...")
            total_draws = len(historical_data)

            # 1. 단일 번호 분석
            for numbers in historical_data:
                for num in numbers:
                    self.number_freq[num - 1] += 1
                    self.section_freq[(num - 1) // 10] += 1

            # 출현 확률 계산
            self.number_prob = self.number_freq / total_draws

            # 2. 번호 쌍 분석
            for numbers in historical_data:
                for i in range(6):
                    for j in range(i + 1, 6):
                        self.pair_freq[numbers[i] - 1, numbers[j] - 1] += 1
                        self.pair_freq[numbers[j] - 1, numbers[i] - 1] += 1

            # 3. 번호 간격 분석
            for numbers in historical_data:
                sorted_nums = sorted(numbers)
                gaps = np.diff(sorted_nums)
                for gap in gaps:
                    self.gap_freq[gap] += 1

            # 4. 연속성 분석
            self.analyze_streaks(historical_data)

            # 5. 최근 트렌드 분석
            self.recent_numbers = historical_data[-10:]  # 최근 10회차

            # 6. 상세 통계 계산
            self.calculate_detailed_stats(historical_data)

            logging.info("통계 분석 완료")
            self.log_analysis_results()

        except Exception as e:
            logging.error(f"데이터 분석 중 오류 발생: {str(e)}")
            raise

    def analyze_streaks(self, historical_data):
        """연속 출현/미출현 패턴 분석"""
        try:
            for num in range(1, 46):
                streak = 0
                max_streak = 0
                for numbers in historical_data:
                    if num in numbers:
                        if streak < 0:
                            self.streak_data[f"미출현_{abs(streak)}"] += 1
                        streak = 1
                    else:
                        if streak > 0:
                            self.streak_data[f"출현_{streak}"] += 1
                        streak = -1
                    max_streak = max(max_streak, abs(streak))

            logging.info(f"최대 연속 기록: {max_streak}회")

        except Exception as e:
            logging.error(f"연속성 분석 중 오류: {str(e)}")
            raise

    def calculate_detailed_stats(self, historical_data):
        """상세 통계 정보 계산"""
        try:
            for num in range(1, 46):
                # 기본 출현 정보
                appearances = self.number_freq[num - 1]
                probability = self.number_prob[num - 1]

                # 최근 출현 정보
                recent_count = sum(1 for numbers in self.recent_numbers
                                   if num in numbers)

                # 번호 조합 정보
                common_pairs = np.argsort(self.pair_freq[num - 1])[-5:]

                self.number_stats[num] = {
                    'total_appearances': int(appearances),
                    'probability': probability,
                    'recent_appearances': recent_count,
                    'common_pairs': [i + 1 for i in common_pairs],
                    'section': (num - 1) // 10
                }

        except Exception as e:
            logging.error(f"상세 통계 계산 중 오류: {str(e)}")
            raise

    def train(self, train_data, validation_data=None):
        """학습 인터페이스 구현"""
        try:
            self.analyze_data(train_data)

            # 통계 모델 상태 저장
            model_state = {
                'number_freq': self.number_freq,
                'pair_freq': self.pair_freq,
                'gap_freq': dict(self.gap_freq),
                'section_freq': self.section_freq,
                'streak_data': dict(self.streak_data),
                'number_stats': self.number_stats
            }

            np.save('best_models/statistical_model.npy', model_state)

        except Exception as e:
            logging.error(f"모델 학습 중 오류: {str(e)}")
            raise

    # def predict_numbers(self, input_data=None, n_predictions=6):
    #     """번호 예측"""
    #     try:
    #         selected_numbers = []
    #         probabilities = self.calculate_probabilities()
    #
    #         while len(selected_numbers) < n_predictions:
    #             # 남은 번호들의 확률 계산
    #             remaining_prob = probabilities.copy()
    #             remaining_prob[np.array(selected_numbers) - 1] = 0
    #             remaining_prob = remaining_prob / np.sum(remaining_prob)
    #
    #             # 번호 선택
    #             selected = np.random.choice(45, p=remaining_prob) + 1
    #
    #             # 선택한 번호와 기존 번호들 간의 관계 검증
    #             if self.validate_selection(selected, selected_numbers):
    #                 selected_numbers.append(selected)
    #
    #         result = self.format_numbers(selected_numbers)
    #         self.log_prediction_results(result, probabilities)
    #         return result
    #
    #     except Exception as e:
    #         logging.error(f"예측 중 오류: {str(e)}")
    #         raise

    def predict_numbers(self, input_data=None, n_predictions=6):
        """번호 예측"""
        try:
            selected_numbers = []
            probabilities = self.calculate_probabilities()

            while len(selected_numbers) < n_predictions:
                # 남은 번호들의 확률 계산
                remaining_prob = probabilities.copy()

                # 이미 선택된 번호들의 인덱스를 정수형으로 변환
                if selected_numbers:
                    selected_indices = [int(num - 1) for num in selected_numbers]
                    remaining_prob[selected_indices] = 0

                # 확률 정규화
                if np.sum(remaining_prob) > 0:
                    remaining_prob = remaining_prob / np.sum(remaining_prob)
                else:
                    raise ValueError("모든 확률이 0입니다.")

                # 번호 선택 (1-based index로 변환)
                selected = int(np.random.choice(45, p=remaining_prob)) + 1

                # 선택한 번호와 기존 번호들 간의 관계 검증
                if self.validate_selection(selected, selected_numbers):
                    selected_numbers.append(selected)

            result = self.format_numbers(selected_numbers)
            self.log_prediction_results(result, probabilities)
            return result

        except Exception as e:
            logging.error(f"예측 중 오류: {str(e)}")
            raise

    # def calculate_probabilities(self):
    #     """각 번호의 출현 확률 계산"""
    #     try:
    #         # 기본 확률 (과거 출현 빈도 기반)
    #         probabilities = self.number_prob.copy()
    #
    #         # 최근 트렌드 반영
    #         recent_prob = np.zeros(45)
    #         for numbers in self.recent_numbers:
    #             for num in numbers:
    #                 recent_prob[num - 1] += 1
    #         recent_prob = recent_prob / len(self.recent_numbers)
    #
    #         # 구간 균형 반영
    #         section_balance = 1 - (self.section_freq / np.sum(self.section_freq))
    #         section_prob = np.zeros(45)
    #         for i in range(45):
    #             section_prob[i] = section_balance[i // 10]
    #
    #         # 최종 확률 계산 (가중치 적용)
    #         final_prob = (0.4 * probabilities +
    #                       0.4 * recent_prob +
    #                       0.2 * section_prob)
    #
    #         return final_prob / np.sum(final_prob)
    #
    #     except Exception as e:
    #         logging.error(f"확률 계산 중 오류: {str(e)}")
    #         raise

    def calculate_probabilities(self):
        """각 번호의 출현 확률 계산"""
        try:
            # 기본 확률 (과거 출현 빈도 기반)
            probabilities = self.number_prob.copy()

            # 최근 트렌드 반영
            recent_prob = np.zeros(45)
            for numbers in self.recent_numbers:
                for num in numbers:
                    recent_prob[int(num - 1)] += 1
            recent_prob = recent_prob / len(self.recent_numbers)

            # 구간 균형 반영
            section_balance = 1 - (self.section_freq / np.sum(self.section_freq))
            section_prob = np.zeros(45)
            for i in range(45):
                section_prob[i] = section_balance[i // 10]

            # 최종 확률 계산 (가중치 적용)
            final_prob = (0.4 * probabilities +
                          0.4 * recent_prob +
                          0.2 * section_prob)

            # 확률 정규화
            if np.sum(final_prob) > 0:
                final_prob = final_prob / np.sum(final_prob)
            else:
                raise ValueError("유효하지 않은 확률 분포")

            return final_prob

        except Exception as e:
            logging.error(f"확률 계산 중 오류: {str(e)}")
            raise


    def validate_selection(self, number, selected_numbers):
        """선택된 번호의 유효성 검증"""
        if not selected_numbers:
            return True

        # 연속된 번호 체크
        for selected in selected_numbers:
            if abs(number - selected) == 1:
                return False

        # 같은 구간에 너무 많은 번호가 있는지 체크
        section = (number - 1) // 10
        section_count = sum(1 for n in selected_numbers
                            if (n - 1) // 10 == section)
        if section_count >= 3:
            return False

        return True

    def log_prediction_results(self, numbers, probabilities):
        """예측 결과 로깅"""
        logging.info("\n=== 예측 결과 ===")
        logging.info(f"선택된 번호: {numbers}")

        # 각 번호의 통계 정보 로깅
        for num in numbers:
            stats = self.number_stats[num]
            logging.info(
                f"\n번호 {num} 통계:"
                f"\n - 전체 출현 횟수: {stats['total_appearances']}"
                f"\n - 출현 확률: {stats['probability']:.4f}"
                f"\n - 최근 출현: {stats['recent_appearances']}회"
                f"\n - 자주 함께 나온 번호: {stats['common_pairs']}"
            )

    def get_statistics_report(self):
        """통계 보고서 생성"""
        try:
            report = {
                'hot_numbers': np.argsort(self.number_freq)[-5:] + 1,
                'cold_numbers': np.argsort(self.number_freq)[:5] + 1,
                'common_pairs': [
                                    (i + 1, j + 1)
                                    for i, j in zip(*np.where(self.pair_freq == np.max(self.pair_freq)))
                                ][:5],
                'section_distribution': self.section_freq / np.sum(self.section_freq),
                'streak_patterns': dict(self.streak_data)
            }

            # DataFrame으로 변환하여 저장
            stats_df = pd.DataFrame.from_dict(self.number_stats, orient='index')
            stats_df.to_csv('statistics_report.csv')

            return report

        except Exception as e:
            logging.error(f"통계 보고서 생성 중 오류: {str(e)}")
            raise

    def log_analysis_results(self):
        """분석 결과 로깅"""
        try:
            logging.info("\n=== 통계 분석 결과 ===")

            # 가장 많이 나온 번호
            hot_numbers = np.argsort(self.number_freq)[-5:]
            logging.info("\n최다 출현 번호:")
            for num in hot_numbers:
                logging.info(f"번호 {num + 1}: {int(self.number_freq[num])}회")

            # 가장 적게 나온 번호
            cold_numbers = np.argsort(self.number_freq)[:5]
            logging.info("\n최소 출현 번호:")
            for num in cold_numbers:
                logging.info(f"번호 {num + 1}: {int(self.number_freq[num])}회")

            # 구간별 분포
            logging.info("\n구간별 분포:")
            for i, freq in enumerate(self.section_freq):
                start = i * 10 + 1
                end = min((i + 1) * 10, 45)
                logging.info(f"{start}-{end}: {freq / np.sum(self.section_freq):.2%}")

        except Exception as e:
            logging.error(f"분석 결과 로깅 중 오류: {str(e)}")
            raise

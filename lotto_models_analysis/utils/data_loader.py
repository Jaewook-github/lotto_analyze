
# import sqlite3
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import logging
#
#
# class DataLoader:
#     """로또 데이터 로딩 및 전처리 클래스"""
#
#     def __init__(self, db_path='../data/lotto.db', start_draw=1, end_draw=None,
#                  sequence_mode="전체 데이터", sequence_length=10):
#         self.db_path = db_path
#         self.start_draw = start_draw
#         self.end_draw = end_draw
#         self.sequence_mode = sequence_mode
#         self.sequence_length = sequence_length
#         self.setup_logging()
#
#     def setup_logging(self):
#         """로깅 설정"""
#         logging.basicConfig(
#             filename=f'logs/data_loader_{datetime.now():%Y%m%d_%H%M%S}.log',
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s',
#             encoding='utf-8'
#         )
#
#     def load_data(self):
#         """DB에서 로또 데이터 로드"""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 query = """
#                     SELECT draw_number, num1, num2, num3, num4, num5, num6, bonus
#                     FROM lotto_results
#                     WHERE draw_number >= ? AND draw_number <= ?
#                     ORDER BY draw_number
#                 """
#
#                 # end_draw가 None이면 최대값 조회
#                 if self.end_draw is None:
#                     max_query = "SELECT MAX(draw_number) FROM lotto_results"
#                     self.end_draw = pd.read_sql_query(max_query, conn).iloc[0, 0]
#
#                 df = pd.read_sql_query(query, conn,
#                                        params=(self.start_draw, self.end_draw))
#
#                 logging.info(f"데이터 로드 완료: {len(df)}회차")
#                 return df
#
#         except Exception as e:
#             logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
#             raise
#
#     def preprocess_data(self, df):
#         """데이터 전처리"""
#         try:
#             # 번호 데이터 추출
#             numbers = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values
#
#             # 원-핫 인코딩
#             one_hot = np.zeros((len(numbers), 45))
#             for i, row in enumerate(numbers):
#                 one_hot[i, row - 1] = 1
#
#             # 시퀀스 데이터 생성
#             if self.sequence_mode == "전체 데이터":
#                 sequence_length = len(one_hot) // 10
#                 X = []
#                 y = []
#                 for i in range(0, len(one_hot) - sequence_length, sequence_length // 2):
#                     X.append(one_hot[i:i + sequence_length])
#                     y.append(one_hot[i + sequence_length - 1])
#             else:
#                 X = []
#                 y = []
#                 for i in range(len(one_hot) - self.sequence_length):
#                     X.append(one_hot[i:i + self.sequence_length])
#                     y.append(one_hot[i + self.sequence_length])
#
#             X = np.array(X, dtype=np.float32)
#             y = np.array(y, dtype=np.float32)
#
#             logging.info(f"데이터 전처리 완료 - X shape: {X.shape}, y shape: {y.shape}")
#             return X, y
#
#         except Exception as e:
#             logging.error(f"데이터 전처리 중 오류 발생: {str(e)}")
#             raise
#
#     def create_marking_patterns(self, numbers):
#         """마킹지 패턴 생성"""
#         try:
#             patterns = np.zeros((len(numbers), 7, 7))
#             for i, row in enumerate(numbers):
#                 for num in row:
#                     x = (num - 1) // 7
#                     y = (num - 1) % 7
#                     patterns[i, x, y] = 1
#
#             patterns = patterns.reshape(-1, 7, 7, 1)
#             logging.info(f"마킹 패턴 생성 완료 - shape: {patterns.shape}")
#             return patterns
#
#         except Exception as e:
#             logging.error(f"마킹 패턴 생성 중 오류 발생: {str(e)}")
#             raise
#
#     def get_data_info(self):
#         """데이터 정보 반환"""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
#
#                 # 전체 회차 수
#                 total_query = "SELECT COUNT(*) FROM lotto_results"
#                 total_draws = cursor.execute(total_query).fetchone()[0]
#
#                 # 선택된 범위의 회차 수
#                 range_query = """
#                     SELECT COUNT(*) FROM lotto_results
#                     WHERE draw_number BETWEEN ? AND ?
#                 """
#                 selected_draws = cursor.execute(range_query,
#                                                 (self.start_draw, self.end_draw)
#                                                 ).fetchone()[0]
#
#                 return {
#                     'total_draws': total_draws,
#                     'selected_draws': selected_draws,
#                     'start_draw': self.start_draw,
#                     'end_draw': self.end_draw,
#                     'sequence_mode': self.sequence_mode,
#                     'sequence_length': self.sequence_length
#                 }
#
#         except Exception as e:
#             logging.error(f"데이터 정보 조회 중 오류 발생: {str(e)}")
#             raise
#
#     def validate_data_range(self):
#         """데이터 범위 유효성 검사"""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
#
#                 # 최소/최대 회차 확인
#                 min_max_query = """
#                     SELECT MIN(draw_number), MAX(draw_number)
#                     FROM lotto_results
#                 """
#                 min_draw, max_draw = cursor.execute(min_max_query).fetchone()
#
#                 if self.start_draw < min_draw:
#                     raise ValueError(f"시작 회차가 너무 작습니다. (최소: {min_draw})")
#
#                 if self.end_draw > max_draw:
#                     raise ValueError(f"종료 회차가 너무 큽니다. (최대: {max_draw})")
#
#                 if self.end_draw < self.start_draw:
#                     raise ValueError("종료 회차가 시작 회차보다 작습니다.")
#
#                 required_length = (self.sequence_length + 1
#                                    if self.sequence_mode == "지정 길이" else 11)
#
#                 if self.end_draw - self.start_draw + 1 < required_length:
#                     raise ValueError(
#                         f"선택된 범위가 너무 짧습니다. (최소 {required_length}회차 필요)"
#                     )
#
#                 return True
#
#         except Exception as e:
#             logging.error(f"데이터 범위 검증 중 오류 발생: {str(e)}")
#             raise

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from .preprocessing import DataPreprocessor


class DataLoader:
    """로또 데이터 로딩 및 전처리 클래스"""

    def __init__(self, db_path='../lotto.db', start_draw=1, end_draw=None,
                 sequence_mode="전체 데이터", sequence_length=10):
        self.db_path = db_path
        self.start_draw = start_draw
        self.end_draw = end_draw
        self.sequence_mode = sequence_mode
        self.sequence_length = sequence_length
        self.preprocessor = DataPreprocessor()
        self.setup_logging()

    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            filename=f'logs/data_loader_{datetime.now():%Y%m%d_%H%M%S}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )

    def load_data(self):
        """DB에서 로또 데이터 로드"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT draw_number, num1, num2, num3, num4, num5, num6, bonus 
                    FROM lotto_results 
                    WHERE draw_number >= ? AND draw_number <= ?
                    ORDER BY draw_number
                """

                # end_draw가 None이면 최대값 조회
                if self.end_draw is None:
                    max_query = "SELECT MAX(draw_number) FROM lotto_results"
                    self.end_draw = pd.read_sql_query(max_query, conn).iloc[0, 0]

                df = pd.read_sql_query(query, conn,
                                       params=(self.start_draw, self.end_draw))

                logging.info(f"데이터 로드 완료: {len(df)}회차")
                return df

        except Exception as e:
            logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise

    # def preprocess_data(self, df):
    #     """데이터 전처리"""
    #     try:
    #         # 번호 데이터 추출
    #         numbers = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values
    #
    #         # 시퀀스 모드에 따른 처리
    #         if self.sequence_mode == "전체 데이터":
    #             processed_data = self.preprocessor.prepare_data_for_training(
    #                 numbers,
    #                 sequence_length=len(numbers) // 10  # 전체 데이터의 1/10
    #             )
    #         else:
    #             processed_data = self.preprocessor.prepare_data_for_training(
    #                 numbers,
    #                 sequence_length=self.sequence_length
    #             )
    #
    #         # 고급 특징 생성
    #         advanced_features = self.preprocessor.create_advanced_features(df)
    #         processed_data['advanced_features'] = advanced_features
    #
    #         logging.info("데이터 전처리 완료")
    #         logging.info(f"시퀀스 shape: {processed_data['sequences'].shape}")
    #         logging.info(f"특징 shape: {processed_data['features'].shape}")
    #         logging.info(f"타겟 shape: {processed_data['targets'].shape}")
    #         logging.info(f"고급 특징 shape: {advanced_features.shape}")
    #
    #         return processed_data['sequences'], processed_data['targets']
    #
    #     except Exception as e:
    #         logging.error(f"데이터 전처리 중 오류 발생: {str(e)}")
    #         raise

    def preprocess_data(self, df):
        """데이터 전처리"""
        try:
            # 번호 데이터 추출 및 정수형으로 변환
            numbers = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values.astype(np.int32)

            # 시퀀스 데이터 생성
            if self.sequence_mode == "전체 데이터":
                sequence_length = len(numbers) // 10
                X = []
                y = []
                for i in range(0, len(numbers) - sequence_length, sequence_length // 2):
                    sequence = numbers[i:i + sequence_length]
                    target = numbers[i + sequence_length - 1]

                    # 시퀀스 원-핫 인코딩
                    sequence_encoded = np.zeros((sequence_length, 45))
                    for j, nums in enumerate(sequence):
                        for num in nums:
                            sequence_encoded[j, int(num - 1)] = 1

                    X.append(sequence_encoded)
                    y.append(target)
            else:
                X = []
                y = []
                for i in range(len(numbers) - self.sequence_length):
                    sequence = numbers[i:i + self.sequence_length]
                    target = numbers[i + self.sequence_length]

                    # 시퀀스 원-핫 인코딩
                    sequence_encoded = np.zeros((self.sequence_length, 45))
                    for j, nums in enumerate(sequence):
                        for num in nums:
                            sequence_encoded[j, int(num - 1)] = 1

                    X.append(sequence_encoded)
                    y.append(target)

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            # 타겟 원-핫 인코딩
            y_encoded = np.zeros((len(y), 45))
            for i, nums in enumerate(y):
                for num in nums:
                    y_encoded[i, int(num - 1)] = 1

            logging.info("데이터 전처리 완료")
            logging.info(f"입력 데이터 shape: {X.shape}")
            logging.info(f"출력 데이터 shape: {y_encoded.shape}")

            return X, y_encoded, numbers[-len(X):]

        except Exception as e:
            logging.error(f"데이터 전처리 중 오류 발생: {str(e)}")
            raise


    def create_marking_patterns(self, numbers):
        """마킹지 패턴 생성"""
        try:
            patterns = np.array([
                self.preprocessor.create_marking_pattern(row)
                for row in numbers
            ])

            patterns = patterns.reshape(-1, 7, 7, 1)
            logging.info(f"마킹 패턴 생성 완료 - shape: {patterns.shape}")

            return patterns

        except Exception as e:
            logging.error(f"마킹 패턴 생성 중 오류 발생: {str(e)}")
            raise

    def get_data_info(self):
        """데이터 정보 반환"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 전체 회차 수
                total_query = "SELECT COUNT(*) FROM lotto_results"
                total_draws = cursor.execute(total_query).fetchone()[0]

                # 선택된 범위의 회차 수
                range_query = """
                    SELECT COUNT(*) FROM lotto_results
                    WHERE draw_number BETWEEN ? AND ?
                """
                selected_draws = cursor.execute(range_query,
                                                (self.start_draw, self.end_draw)
                                                ).fetchone()[0]

                return {
                    'total_draws': total_draws,
                    'selected_draws': selected_draws,
                    'start_draw': self.start_draw,
                    'end_draw': self.end_draw,
                    'sequence_mode': self.sequence_mode,
                    'sequence_length': self.sequence_length,
                    'preprocessor_stats': self.preprocessor.feature_statistics
                }

        except Exception as e:
            logging.error(f"데이터 정보 조회 중 오류 발생: {str(e)}")
            raise

    def get_latest_data(self, n_records=10):
        """최근 n회차 데이터 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM lotto_results
                    ORDER BY draw_number DESC
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(n_records,))
                return df.iloc[::-1]  # 오름차순으로 변경

        except Exception as e:
            logging.error(f"최근 데이터 조회 중 오류 발생: {str(e)}")
            raise

    def prepare_prediction_data(self, latest_draws=10):
        """예측을 위한 데이터 준비"""
        try:
            # 최근 데이터 로드
            recent_data = self.get_latest_data(latest_draws)
            numbers = recent_data[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values

            # 예측용 데이터 준비
            prediction_data = self.preprocessor.prepare_data_for_prediction(
                numbers,
                sequence_length=self.sequence_length
            )

            # 고급 특징 추가
            advanced_features = self.preprocessor.create_advanced_features(recent_data)
            prediction_data['advanced_features'] = advanced_features.iloc[-1:]

            return prediction_data

        except Exception as e:
            logging.error(f"예측 데이터 준비 중 오류 발생: {str(e)}")
            raise

    def validate_data_range(self):
        """데이터 범위 유효성 검사"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 최소/최대 회차 확인
                min_max_query = """
                    SELECT MIN(draw_number), MAX(draw_number) 
                    FROM lotto_results
                """
                min_draw, max_draw = cursor.execute(min_max_query).fetchone()

                if self.start_draw < min_draw:
                    raise ValueError(f"시작 회차가 너무 작습니다. (최소: {min_draw})")

                if self.end_draw > max_draw:
                    raise ValueError(f"종료 회차가 너무 큽니다. (최대: {max_draw})")

                if self.end_draw < self.start_draw:
                    raise ValueError("종료 회차가 시작 회차보다 작습니다.")

                required_length = (self.sequence_length + 1
                                   if self.sequence_mode == "지정 길이" else 11)

                if self.end_draw - self.start_draw + 1 < required_length:
                    raise ValueError(
                        f"선택된 범위가 너무 짧습니다. (최소 {required_length}회차 필요)"
                    )

                return True

        except Exception as e:
            logging.error(f"데이터 범위 검증 중 오류 발생: {str(e)}")
            raise
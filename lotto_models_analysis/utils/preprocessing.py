
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging


class DataPreprocessor:
    """로또 데이터 전처리 클래스"""

    def __init__(self):
        self.num_scaler = MinMaxScaler()
        self.seq_scaler = StandardScaler()
        self.feature_statistics = {}

    def create_sequences(self, numbers, sequence_length=10):
        """시계열 시퀀스 생성"""
        sequences = []
        targets = []

        for i in range(len(numbers) - sequence_length):
            seq = numbers[i:i + sequence_length]
            target = numbers[i + sequence_length]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def encode_numbers(self, numbers):
        """번호 원-핫 인코딩"""
        encoded = np.zeros((len(numbers), 45))
        for i, row in enumerate(numbers):
            encoded[i, row - 1] = 1
        return encoded

    def create_marking_pattern(self, numbers):
        """마킹지 패턴 생성"""
        pattern = np.zeros((7, 7))
        for num in numbers:
            x = (num - 1) // 7
            y = (num - 1) % 7
            pattern[x, y] = 1
        return pattern

    def calculate_number_stats(self, numbers):
        """번호 통계 계산"""
        stats = {}
        numbers_flat = numbers.flatten()

        # 기본 통계
        stats['mean'] = np.mean(numbers_flat)
        stats['std'] = np.std(numbers_flat)
        stats['min'] = np.min(numbers_flat)
        stats['max'] = np.max(numbers_flat)

        # 구간별 분포
        sections = np.zeros(5)
        for num in numbers_flat:
            sections[(num - 1) // 10] += 1
        stats['section_dist'] = sections / len(numbers_flat)

        return stats

    def analyze_patterns(self, numbers):
        """번호 패턴 분석"""
        patterns = {
            'consecutive': 0,  # 연속된 번호
            'even_odd': [],  # 홀짝 비율
            'section_dist': []  # 구간 분포
        }

        for row in numbers:
            # 연속된 번호 확인
            diffs = np.diff(sorted(row))
            patterns['consecutive'] += np.sum(diffs == 1)

            # 홀짝 비율
            evens = np.sum(row % 2 == 0)
            patterns['even_odd'].append(evens / len(row))

            # 구간 분포
            sections = np.zeros(5)
            for num in row:
                sections[(num - 1) // 10] += 1
            patterns['section_dist'].append(sections / len(row))

        patterns['even_odd'] = np.mean(patterns['even_odd'])
        patterns['section_dist'] = np.mean(patterns['section_dist'], axis=0)

        return patterns

    def extract_features(self, numbers):
        """특징 추출"""
        features = []

        for row in numbers:
            row_features = []
            sorted_nums = np.sort(row)

            # 기본 통계
            row_features.extend([
                np.mean(sorted_nums),
                np.std(sorted_nums),
                np.max(sorted_nums) - np.min(sorted_nums)
            ])

            # 간격 특징
            gaps = np.diff(sorted_nums)
            row_features.extend([
                np.mean(gaps),
                np.std(gaps),
                np.max(gaps)
            ])

            # 구간 분포
            sections = np.zeros(5)
            for num in sorted_nums:
                sections[(num - 1) // 10] += 1
            row_features.extend(sections / len(row))

            # 홀짝 비율
            evens = np.sum(row % 2 == 0)
            row_features.append(evens / len(row))

            features.append(row_features)

        return np.array(features)

    def create_advanced_features(self, df):
        """고급 특징 생성"""
        features = {}

        # 이동 평균
        for window in [3, 5, 10]:
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                features[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()

        # 이동 표준편차
        for window in [5, 10]:
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                features[f'{col}_std_{window}'] = df[col].rolling(window=window).std()

        # 번호 간 차이
        for i in range(1, 6):
            curr_col = f'num{i}'
            next_col = f'num{i + 1}'
            features[f'diff_{i}'] = df[next_col] - df[curr_col]

        return pd.DataFrame(features)

    def normalize_features(self, features, train=True):
        """특징 정규화"""
        if train:
            return self.num_scaler.fit_transform(features)
        return self.num_scaler.transform(features)

    def normalize_sequences(self, sequences, train=True):
        """시퀀스 정규화"""
        shape = sequences.shape
        flattened = sequences.reshape(-1, shape[-1])

        if train:
            normalized = self.seq_scaler.fit_transform(flattened)
        else:
            normalized = self.seq_scaler.transform(flattened)

        return normalized.reshape(shape)

    def prepare_data_for_training(self, numbers, sequence_length=10):
        """학습용 데이터 준비"""
        try:
            # 기본 통계 계산
            self.feature_statistics = self.calculate_number_stats(numbers)

            # 패턴 분석
            pattern_info = self.analyze_patterns(numbers)
            self.feature_statistics.update(pattern_info)

            # 시퀀스 생성
            X_seq, y = self.create_sequences(numbers, sequence_length)

            # 특징 추출
            X_features = self.extract_features(numbers[sequence_length:])

            # 데이터 정규화
            X_seq_norm = self.normalize_sequences(X_seq, train=True)
            X_features_norm = self.normalize_features(X_features, train=True)

            # 원-핫 인코딩
            y_encoded = self.encode_numbers(y)

            logging.info("데이터 전처리 완료")
            logging.info(f"시퀀스 shape: {X_seq_norm.shape}")
            logging.info(f"특징 shape: {X_features_norm.shape}")
            logging.info(f"타겟 shape: {y_encoded.shape}")

            return {
                'sequences': X_seq_norm,
                'features': X_features_norm,
                'targets': y_encoded,
                'stats': self.feature_statistics
            }

        except Exception as e:
            logging.error(f"데이터 준비 중 오류 발생: {str(e)}")
            raise

    def prepare_data_for_prediction(self, numbers, sequence_length=10):
        """예측용 데이터 준비"""
        try:
            # 마지막 시퀀스 추출
            last_sequence = numbers[-sequence_length:]

            # 특징 추출
            features = self.extract_features(numbers[-1:])

            # 데이터 정규화
            seq_norm = self.normalize_sequences(
                last_sequence.reshape(1, sequence_length, -1),
                train=False
            )
            features_norm = self.normalize_features(features, train=False)

            # 마킹 패턴 생성
            marking_pattern = self.create_marking_pattern(numbers[-1])

            return {
                'sequence': seq_norm,
                'features': features_norm,
                'marking_pattern': marking_pattern
            }

        except Exception as e:
            logging.error(f"예측 데이터 준비 중 오류 발생: {str(e)}")
            raise

    def inverse_transform_predictions(self, predictions):
        """예측 결과 역변환"""
        try:
            # 확률을 번호로 변환
            numbers = []
            for pred in predictions:
                # 상위 6개 확률을 가진 인덱스 선택
                top_indices = np.argsort(pred)[-6:]
                numbers.append(sorted(top_indices + 1))

            return numbers

        except Exception as e:
            logging.error(f"예측 결과 변환 중 오류 발생: {str(e)}")
            raise

from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
import os
from pathlib import Path
import threading
import queue
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행하기 위한 설정
import matplotlib.pyplot as plt
import io
import base64
from config import Config
import logging
import sqlite3
from sqlite3 import Error
import copy
import pickle
import traceback
import sys
import locale

sys.stdout.reconfigure(encoding='utf-8')

# Flask 앱 초기화
app = Flask(__name__)
Config.init_app(app)

# 전역 변수
task_queue = queue.Queue()
results = {}
active_tasks = {}

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / 'webapp.log', encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout) # stdout에 대한 인코딩 설정
    ]
)
logger = logging.getLogger(__name__)

# 폰트 설정
try:
    if sys.platform == 'win32':
        font_path = 'C:/Windows/Fonts/malgun.ttf'
        plt.rcParams['font.family'] = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
    else:
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.family'] = 'sans-serif'


class ModelState:
    """모델 상태를 관리하는 기본 클래스"""

    def __init__(self, numbers_memory=None, number_stats=None, score=0):
        self.numbers_memory = numbers_memory if numbers_memory is not None else {}
        self.number_stats = number_stats if number_stats is not None else {}
        self.score = score

    def copy(self):
        return ModelState(
            numbers_memory=copy.deepcopy(self.numbers_memory),
            number_stats=copy.deepcopy(self.number_stats),
            score=self.score
        )

    def save_to_file(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'numbers_memory': self.numbers_memory,
                'number_stats': self.number_stats,
                'score': self.score
            }, f)

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return cls(
                numbers_memory=data.get('numbers_memory', {}),
                number_stats=data.get('number_stats', {}),
                score=data.get('score', 0)
            )


class DatabaseManager:
    """데이터베이스 관리 클래스"""

    def __init__(self):
        self.db_path = Config.DATABASE_PATH

    def connect(self):
        try:
            return sqlite3.connect(self.db_path)
        except Error as e:
            logger.error("Database connection error", exc_info=True)
            raise

    def get_historical_data(self, limit: int = None) -> pd.DataFrame:
        try:
            base_query = """
                SELECT draw_number, num1, num2, num3, num4, num5, num6, bonus,
                       money1, money2, money3, money4, money5
                FROM lotto_results
            """

            if limit and str(limit).isdigit() and int(limit) > 0:
                query = f"""
                    {base_query}
                    WHERE draw_number IN (
                        SELECT draw_number 
                        FROM lotto_results 
                        ORDER BY draw_number DESC 
                        LIMIT {limit}
                    )
                    ORDER BY draw_number ASC
                """
            else:
                query = f"{base_query} ORDER BY draw_number ASC"

            with self.connect() as conn:
                logger.info(f"Retrieving {'all' if limit is None else limit} historical data")
                df = pd.read_sql_query(query, conn)
                logger.info(f"Retrieved {len(df)} records")
                return df

        except Exception as e:
            logger.error("Data retrieval error", exc_info=True)
            raise


class LearningAnalyzer:
    """학습 과정 분석 클래스"""

    def __init__(self):
        self.learning_results = {}
        self.prize_counts = {}
        self.best_model = None
        self.best_score = 0

    def evaluate_model(self, analyzer, actual_numbers, iterations=100):
        total_score = 0
        match_counts = {i: 0 for i in range(7)}

        for _ in range(iterations):
            predicted = analyzer.select_numbers()
            matches = len(set(predicted) & set(actual_numbers))
            total_score += self._calculate_match_score(matches)
            match_counts[matches] += 1

        average_score = total_score / iterations
        return average_score, match_counts

    def _calculate_match_score(self, matches):
        score_table = {
            6: 1000,  # 1등
            5: 50,  # 2등/3등
            4: 20,  # 4등
            3: 5,  # 5등
            2: 1,  # 미당첨이지만 약간의 가치
            1: 0,  # 거의 무가치
            0: 0  # 완전 무가치
        }
        return score_table.get(matches, 0)

    def analyze_match(self, draw_number, actual_numbers, predicted_numbers):
        matches = set(predicted_numbers) & set(actual_numbers)
        match_count = len(matches)
        prize_rank = self._get_prize_rank(match_count)

        if draw_number not in self.prize_counts:
            self.prize_counts[draw_number] = {i: 0 for i in range(6)}

        self.prize_counts[draw_number][prize_rank] += 1

        return {
            'draw_number': draw_number,
            'actual_numbers': actual_numbers,
            'predicted_numbers': predicted_numbers,
            'matches': list(matches),
            'match_count': match_count,
            'prize_rank': prize_rank
        }

    def _get_prize_rank(self, match_count):
        prize_ranks = {
            6: 1,  # 1등
            5: 2,  # 2등
            4: 3,  # 3등
            3: 4,  # 4등
            2: 5  # 5등
        }
        return prize_ranks.get(match_count, 0)


class LottoAnalyzer:
    """로또 번호 분석 및 예측 클래스"""

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.numbers_memory = {i: 1.0 for i in range(1, 46)}
        self.number_stats = {i: 0 for i in range(1, 46)}

    def get_state(self) -> ModelState:
        return ModelState(
            numbers_memory=dict(self.numbers_memory),
            number_stats=dict(self.number_stats)
        )

    def set_state(self, state: ModelState):
        self.numbers_memory = copy.deepcopy(state.numbers_memory)
        self.number_stats = copy.deepcopy(state.number_stats)

    def analyze_historical_data(self, df: pd.DataFrame):
        total_records = len(df)

        for _, row in df.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 7)]
            for num in numbers:
                self.number_stats[num] += 1

    def select_numbers(self) -> list:
        weights = list(self.numbers_memory.values())
        weight_sum = sum(weights)
        probabilities = [w / weight_sum for w in weights]

        selected = []
        while len(selected) < 6:
            num = np.random.choice(range(1, 46), p=probabilities)
            if num not in selected:
                selected.append(num)
                probabilities[int(num) - 1] *= 0.5
                probabilities = [p / sum(probabilities) for p in probabilities]

        return sorted(selected)

    def update_weights(self, numbers: list, score: int):
        for num in numbers:
            old_weight = self.numbers_memory[num]
            increase = score * self.learning_rate

            frequency = self.number_stats.get(num, 0)
            max_frequency = max(self.number_stats.values())
            frequency_factor = frequency / max_frequency if max_frequency > 0 else 0

            self.numbers_memory[num] += increase * (1 + frequency_factor)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_prediction', methods=['POST'])
def start_prediction():
    try:
        data = request.json
        games = int(data['games'])
        learning_rate = float(data['learning_rate'])
        iterations = int(data['iterations'])
        draw_limit = None if data['learning_draws'] == "전체" else int(data['learning_draws'])

        # 입력값 검증
        if games > 50:
            return jsonify({'error': '최대 50게임까지만 가능합니다'}), 400
        if learning_rate <= 0 or learning_rate > 1:
            return jsonify({'error': '학습률은 0과 1 사이여야 합니다'}), 400
        if iterations <= 0:
            return jsonify({'error': '학습 반복 횟수는 1 이상이어야 합니다'}), 400

        # 작업 ID 생성
        task_id = datetime.now().strftime('%Y%m%d%H%M%S')

        # 작업 초기화
        results[task_id] = {
            'status': 'initializing',
            'progress': 0,
            'log_messages': [],
            'predictions': [],
            'graphs': None
        }

        # 예측 작업 시작
        thread = threading.Thread(
            target=run_prediction,
            args=(task_id, games, learning_rate, iterations, draw_limit)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'task_id': task_id,
            'message': '예측이 시작되었습니다.'
        })

    except Exception as e:
        logger.error(f"Prediction start error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/help')
def help_page():
    return render_template('help.html')


@app.route('/get_progress/<task_id>')
def get_progress(task_id):
    if task_id not in results:
        return jsonify({'error': 'Task not found'}), 404

    # NumPy 타입을 Python 기본 타입으로 변환
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # 결과 딕셔너리의 모든 값을 변환
    serializable_results = {}
    for key, value in results[task_id].items():
        if isinstance(value, dict):
            serializable_results[key] = {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            serializable_results[key] = convert_to_serializable(value)

    return jsonify(serializable_results)


@app.route('/get_graphs/<task_id>')
def get_graphs(task_id):
    if task_id not in results or 'graphs' not in results[task_id]:
        return jsonify({'error': 'Graphs not available'}), 404
    return jsonify(results[task_id]['graphs'])


@app.route('/download_results/<task_id>')
def download_results(task_id):
    if task_id not in results or not results[task_id].get('file_path'):
        return jsonify({'error': 'Results not available'}), 404

    try:
        return send_file(
            results[task_id]['file_path'],
            as_attachment=True,
            download_name=f'lotto_prediction_{task_id}.xlsx'
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Download failed'}), 500


# def run_prediction(task_id, games, learning_rate, iterations, draw_limit):
#     """예측 실행 함수"""
#     try:
#         # 데이터베이스 매니저 초기화
#         db_manager = DatabaseManager()
#
#         # 분석기 초기화
#         analyzer = LottoAnalyzer(learning_rate)
#         learning_analyzer = LearningAnalyzer()
#
#         # 상태 업데이트
#         results[task_id] = {
#             'status': 'running',
#             'progress': 0,
#             'log_messages': ['데이터 로딩 중...']
#         }
#
#         # 데이터 로드
#         historical_data = db_manager.get_historical_data(draw_limit)
#
#         overall_best_state = None
#         overall_best_score = 0.0
#         total_draws = len(historical_data)
#
#         # 각 회차별 학습
#         for idx, row in enumerate(historical_data.iterrows(), 1):
#             _, data = row
#             draw_number = int(data['draw_number'])
#             actual_numbers = [int(data[f'num{i}']) for i in range(1, 7)]
#
#             results[task_id]['log_messages'].append(f"\n=== {draw_number}회차 학습 ===")
#             results[task_id]['log_messages'].append(f"실제 당첨번호: {actual_numbers}")
#
#             # 현재 회차 학습
#             best_state = None
#             best_score = 0.0
#             match_summary = {i: 0 for i in range(7)}
#
#             for iteration in range(iterations):
#                 predicted_numbers = analyzer.select_numbers()
#                 matches = len(set(predicted_numbers) & set(actual_numbers))
#                 match_summary[matches] += 1
#
#                 current_score, _ = learning_analyzer.evaluate_model(
#                     analyzer, actual_numbers, iterations=10
#                 )
#                 current_score = float(current_score)
#
#                 if current_score > best_score:
#                     best_score = float(current_score)
#                     best_state = analyzer.get_state()
#                     best_state.score = best_score
#
#                     if current_score > overall_best_score:
#                         overall_best_score = float(current_score)
#                         overall_best_state = best_state.copy()
#
#                 if matches > 0:
#                     analyzer.update_weights(
#                         list(set(predicted_numbers) & set(actual_numbers)),
#                         int(matches)
#                     )
#
#             # 회차별 결과 요약
#             summary_text = (
#                 f"학습 완료:\n"
#                 f"- 최고 점수: {best_score:.2f}\n"
#                 f"- 매칭 통계:\n"
#             )
#
#             for matches, count in match_summary.items():
#                 if count > 0:
#                     percentage = float(count / iterations * 100)
#                     summary_text += f"  {matches}개 일치: {count}회 ({percentage:.2f}%)\n"
#
#             results[task_id]['log_messages'].append(summary_text)
#
#             # 진행률 업데이트
#             progress = float((idx / total_draws) * 100)
#             results[task_id]['progress'] = progress
#
#         # 최종 예측 수행
#         if overall_best_state:
#             analyzer.set_state(overall_best_state)
#             final_predictions = []
#
#             for _ in range(games):
#                 predicted_numbers = [int(x) for x in analyzer.select_numbers()]
#                 final_predictions.append(predicted_numbers)
#                 results[task_id]['predictions'] = final_predictions
#
#             # 그래프 생성
#             results[task_id]['graphs'] = generate_plots(analyzer, historical_data)
#
#             # 결과 저장
#             file_path = save_results(task_id, final_predictions)
#             results[task_id]['file_path'] = str(file_path)
#
#             # 완료 상태 업데이트
#             results[task_id].update({
#                 'status': 'completed',
#                 'score': float(overall_best_score),
#                 'log_messages': results[task_id]['log_messages'] + [
#                     f"\n예측 완료!\n"
#                     f"최고 성능 점수: {overall_best_score:.2f}\n"
#                     f"생성된 게임 수: {games}"
#                 ]
#             })
#
#     except Exception as e:
#         logger.error(f"Prediction process error: {str(e)}", exc_info=True)
#         results[task_id].update({
#             'status': 'error',
#             'error': str(e),
#             'log_messages': results[task_id].get('log_messages', []) + [f"오류 발생: {str(e)}"]
#         })
#
#     # 모든 숫자 타입을 Python 기본 타입으로 변환
#     def convert_values(d):
#         if isinstance(d, dict):
#             return {k: convert_values(v) for k, v in d.items()}
#         elif isinstance(d, list):
#             return [convert_values(x) for x in d]
#         elif isinstance(d, (np.integer, np.int32, np.int64)):
#             return int(d)
#         elif isinstance(d, (np.floating, np.float32, np.float64)):
#             return float(d)
#         elif isinstance(d, np.ndarray):
#             return d.tolist()
#         return d
#
#     # 결과 딕셔너리의 모든 값을 변환
#     results[task_id] = convert_values(results[task_id])

def run_prediction(task_id, games, learning_rate, iterations, draw_limit):
    """예측 실행 함수"""
    try:
        # 데이터베이스 매니저 초기화
        db_manager = DatabaseManager()

        # 분석기 초기화
        analyzer = LottoAnalyzer(learning_rate)
        learning_analyzer = LearningAnalyzer()

        # 상태 업데이트
        results[task_id] = {
            'status': 'running',
            'progress': 0,
            'log_messages': ['데이터 로딩 중...']
        }

        # 데이터 로드
        historical_data = db_manager.get_historical_data(draw_limit)
        data_range = "전체" if not draw_limit else f"최근 {draw_limit}회차"

        logger.info(f"데이터 로드 완료: {data_range}")
        results[task_id]['log_messages'].append(f"데이터 로드 완료: {data_range}")

        overall_best_state = None
        overall_best_score = 0.0
        total_draws = len(historical_data)

        # 각 회차별 학습
        for idx, row in enumerate(historical_data.iterrows(), 1):
            _, data = row
            draw_number = int(data['draw_number'])
            actual_numbers = [int(data[f'num{i}']) for i in range(1, 7)]

            results[task_id]['log_messages'].append(f"\n=== {draw_number}회차 학습 ===")
            results[task_id]['log_messages'].append(f"실제 당첨번호: {actual_numbers}")

            # 현재 회차 학습
            best_state = None
            best_score = 0.0
            match_summary = {i: 0 for i in range(7)}

            for iteration in range(iterations):
                predicted_numbers = analyzer.select_numbers()
                matches = len(set(predicted_numbers) & set(actual_numbers))
                match_summary[matches] += 1

                current_score, _ = learning_analyzer.evaluate_model(
                    analyzer, actual_numbers, iterations=10
                )
                current_score = float(current_score)

                if current_score > best_score:
                    best_score = float(current_score)
                    best_state = analyzer.get_state()
                    best_state.score = best_score

                    if current_score > overall_best_score:
                        overall_best_score = float(current_score)
                        overall_best_state = best_state.copy()

                if matches > 0:
                    analyzer.update_weights(
                        list(set(predicted_numbers) & set(actual_numbers)),
                        int(matches)
                    )

            # 회차별 결과 요약
            summary_text = (
                f"학습 완료:\n"
                f"- 최고 점수: {best_score:.2f}\n"
                f"- 매칭 통계:\n"
            )

            for matches, count in match_summary.items():
                if count > 0:
                    percentage = float(count / iterations * 100)
                    summary_text += f"  {matches}개 일치: {count}회 ({percentage:.2f}%)\n"

            results[task_id]['log_messages'].append(summary_text)

            # 진행률 업데이트
            progress = float((idx / total_draws) * 100)
            results[task_id]['progress'] = progress

            # 분석기의 number_stats 업데이트
            for num in actual_numbers:
                analyzer.number_stats[num] = analyzer.number_stats.get(num, 0) + 1

        # 최종 예측 수행
        if overall_best_state:
            analyzer.set_state(overall_best_state)
            final_predictions = []

            for _ in range(games):
                predicted_numbers = [int(x) for x in analyzer.select_numbers()]
                final_predictions.append(predicted_numbers)
                results[task_id]['predictions'] = final_predictions

            # 그래프 생성
            logger.info("Generating graphs...")
            graphs = generate_plots(analyzer, historical_data)
            if graphs:
                logger.info("Graphs generated successfully")
                results[task_id]['graphs'] = graphs
            else:
                logger.error("Failed to generate graphs")

            # 결과 저장
            file_path = save_results(task_id, final_predictions)
            results[task_id]['file_path'] = str(file_path)

            # 완료 상태 업데이트
            results[task_id].update({
                'status': 'completed',
                'score': float(overall_best_score),
                'log_messages': results[task_id]['log_messages'] + [
                    f"\n예측 완료!\n"
                    f"최고 성능 점수: {overall_best_score:.2f}\n"
                    f"생성된 게임 수: {games}"
                ]
            })

            logger.info(f"Prediction completed - Best score: {overall_best_score:.2f}")

    except Exception as e:
        logger.error(f"Prediction process error: {str(e)}", exc_info=True)
        results[task_id].update({
            'status': 'error',
            'error': str(e),
            'log_messages': results[task_id].get('log_messages', []) + [f"오류 발생: {str(e)}"]
        })

    # 모든 숫자 타입을 Python 기본 타입으로 변환
    def convert_values(d):
        if isinstance(d, dict):
            return {k: convert_values(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_values(x) for x in d]
        elif isinstance(d, (np.integer, np.int32, np.int64)):
            return int(d)
        elif isinstance(d, (np.floating, np.float32, np.float64)):
            return float(d)
        elif isinstance(d, np.ndarray):
            return d.tolist()
        return d

    # 결과 딕셔너리의 모든 값을 변환
    results[task_id] = convert_values(results[task_id])


# def generate_plots(analyzer, historical_data):
#     """분석 그래프 생성"""
#     try:
#         plots = {}
#
#         # 번호별 출현 빈도 그래프
#         plt.figure(figsize=(12, 6))
#         numbers = list(range(1, 46))
#         frequencies = [analyzer.number_stats[n] for n in numbers]
#         plt.bar(numbers, frequencies)
#         plt.title('번호별 출현 빈도')
#         plt.xlabel('번호')
#         plt.ylabel('출현 횟수')
#         plt.grid(True, linestyle='--', alpha=0.7)
#
#         # 이미지로 변환
#         freq_buf = io.BytesIO()
#         plt.savefig(freq_buf, format='png', bbox_inches='tight')
#         freq_buf.seek(0)
#         plots['frequency'] = base64.b64encode(freq_buf.getvalue()).decode('utf-8')
#         plt.close()
#
#         # 당첨금 트렌드 그래프
#         plt.figure(figsize=(12, 6))
#         draws = historical_data['draw_number']
#         prizes = historical_data['money1']
#         plt.plot(draws, prizes)
#         plt.title('1등 당첨금 트렌드')
#         plt.xlabel('회차')
#         plt.ylabel('당첨금')
#         plt.grid(True, linestyle='--', alpha=0.7)
#
#         # 이미지로 변환
#         trend_buf = io.BytesIO()
#         plt.savefig(trend_buf, format='png', bbox_inches='tight')
#         trend_buf.seek(0)
#         plots['trend'] = base64.b64encode(trend_buf.getvalue()).decode('utf-8')
#         plt.close()
#
#         return plots
#
#     except Exception as e:
#         logger.error(f"Plot generation error: {str(e)}", exc_info=True)
#         return None

def generate_plots(analyzer, historical_data):
    """분석 그래프 생성"""
    try:
        plots = {}

        # 1. 번호별 출현 빈도 그래프
        plt.figure(figsize=(12, 6))
        numbers = list(range(1, 46))
        frequencies = [analyzer.number_stats[n] for n in numbers]

        # 색상 설정
        colors = []
        for n in numbers:
            if 1 <= n <= 10:
                colors.append('#fbc400')  # 노란색
            elif 11 <= n <= 20:
                colors.append('#69c8f2')  # 파란색
            elif 21 <= n <= 30:
                colors.append('#ff7272')  # 빨간색
            elif 31 <= n <= 40:
                colors.append('#aaaaaa')  # 회색
            else:
                colors.append('#b0d840')  # 녹색

        plt.bar(numbers, frequencies, color=colors)
        plt.title('번호별 출현 빈도')
        plt.xlabel('번호')
        plt.ylabel('출현 횟수')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 이미지로 변환
        freq_buf = io.BytesIO()
        plt.savefig(freq_buf, format='png', bbox_inches='tight', dpi=100)
        freq_buf.seek(0)
        plots['frequency'] = base64.b64encode(freq_buf.getvalue()).decode('utf-8')
        plt.close()

        # 2. 당첨금 트렌드 그래프
        plt.figure(figsize=(12, 6))
        draws = historical_data['draw_number']
        prizes = historical_data['money1']
        plt.plot(draws, prizes, color='#2196F3')
        plt.title('1등 당첨금 트렌드')
        plt.xlabel('회차')
        plt.ylabel('당첨금')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ticklabel_format(style='plain', axis='y')

        # 이미지로 변환
        trend_buf = io.BytesIO()
        plt.savefig(trend_buf, format='png', bbox_inches='tight', dpi=100)
        trend_buf.seek(0)
        plots['trend'] = base64.b64encode(trend_buf.getvalue()).decode('utf-8')
        plt.close()

        return plots

    except Exception as e:
        logger.error(f"Plot generation error: {str(e)}", exc_info=True)
        return None


def save_results(task_id, predictions):
    """예측 결과 저장"""
    try:
        # 결과 파일 경로 생성
        file_path = Config.PREDICTIONS_DIR / f'lotto_prediction_{task_id}.xlsx'

        # DataFrame 생성 및 저장
        df = pd.DataFrame(predictions, columns=[f'번호{i + 1}' for i in range(6)])

        with pd.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name='예측번호', index=False)

        return str(file_path)

    except Exception as e:
        logger.error(f"Results saving error: {str(e)}", exc_info=True)
        return None


if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)

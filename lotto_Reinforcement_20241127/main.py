import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
from datetime import datetime
import os
import threading
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import sys
import matplotlib
import matplotlib.font_manager as fm
import copy
import pickle


class ModelState:
    """모델 상태를 관리하는 기본 클래스"""

    def __init__(self, numbers_memory=None, number_stats=None, score=0):
        """
        ModelState 초기화

        Args:
            numbers_memory (dict): 번호별 가중치 정보
            number_stats (dict): 번호별 통계 정보
            score (float): 모델 점수
        """
        self.numbers_memory = numbers_memory if numbers_memory is not None else {}
        self.number_stats = number_stats if number_stats is not None else {}
        self.score = score

    def copy(self):
        """현재 상태의 깊은 복사본을 반환"""
        return ModelState(
            numbers_memory=copy.deepcopy(self.numbers_memory),
            number_stats=copy.deepcopy(self.number_stats),
            score=self.score
        )

    def save_to_file(self, filepath):
        """현재 상태를 파일로 저장"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'numbers_memory': self.numbers_memory,
                'number_stats': self.number_stats,
                'score': self.score
            }, f)

    @classmethod
    def load_from_file(cls, filepath):
        """파일에서 상태를 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return cls(
                numbers_memory=data.get('numbers_memory', {}),
                number_stats=data.get('number_stats', {}),
                score=data.get('score', 0)
            )


class FileManager:
    """파일 및 디렉토리 관리 클래스"""

    def __init__(self):
        self.base_dir = Path('.')
        self.logs_dir = self.base_dir / 'logs'
        self.predictions_dir = self.base_dir / 'predictions'
        self.models_dir = self.base_dir / 'models'
        self.setup_directories()

    def setup_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [self.logs_dir, self.predictions_dir, self.models_dir]:
            directory.mkdir(exist_ok=True)

    def get_new_log_file(self) -> Path:
        """새로운 로그 파일 경로 생성"""
        current_date = datetime.now().strftime('%Y%m%d')

        # 해당 날짜의 기존 로그 파일들 확인
        existing_logs = list(self.logs_dir.glob(f'lotto_prediction_{current_date}-*.log'))

        # 새로운 번호 결정
        if not existing_logs:
            new_number = 1
        else:
            max_number = max([int(log.stem.split('-')[-1]) for log in existing_logs])
            new_number = max_number + 1

        return self.logs_dir / f'lotto_prediction_{current_date}-{new_number}.log'

    def get_prediction_file(self, extension: str) -> Path:
        """예측 결과 파일 경로 반환"""
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.predictions_dir / f'lotto_prediction_{current_datetime}.{extension}'

    def get_model_file(self) -> Path:
        """모델 저장 파일 경로 반환"""
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.models_dir / f'best_model_{current_datetime}.pkl'


class LogManager:
    """로깅 관리 클래스"""

    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.setup_logging()

    def setup_logging(self):
        """로깅 설정"""
        log_file = self.file_manager.get_new_log_file()

        logger = logging.getLogger('LottoPrediction')
        logger.setLevel(logging.INFO)  # DEBUG에서 INFO로 변경

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger.handlers = []
        logger.addHandler(file_handler)
        self.logger = logger

    def log_info(self, message: str):
        """정보 로깅"""
        self.logger.info(message)

    def log_error(self, message: str, exc_info=None):
        """에러 로깅"""
        if exc_info:
            self.logger.error(message, exc_info=exc_info)
        else:
            self.logger.error(message)

    def log_debug(self, message: str):
        """디버그 정보 로깅"""
        if not message.startswith("Selected numbers"):  # Selected numbers 로깅 제외
            self.logger.debug(message)


class DatabaseManager:
    """데이터베이스 관리 클래스"""

    def __init__(self, db_path: str, log_manager: LogManager):
        self.db_path = db_path
        self.log_manager = log_manager
        self.connection = None

    def connect(self):
        """데이터베이스 연결"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.log_manager.log_info("Database connected successfully")
            return self.connection
        except Error as e:
            self.log_manager.log_error("Database connection error", exc_info=True)
            raise

    def get_historical_data(self, limit: int = None) -> pd.DataFrame:
        """당첨 이력 조회"""
        try:
            base_query = """
                SELECT draw_number, num1, num2, num3, num4, num5, num6, bonus,
                       money1, money2, money3, money4, money5
                FROM lotto_results
            """

            if limit and str(limit).isdigit() and int(limit) > 0:
                # 최근 N회차 데이터만 조회하는 서브쿼리
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
                # 전체 데이터 조회
                query = f"{base_query} ORDER BY draw_number ASC"

            with self.connect() as conn:
                data_count = "전체" if not limit else limit
                self.log_manager.log_info(f"Retrieving {data_count} historical data")
                df = pd.read_sql_query(query, conn)
                self.log_manager.log_info(f"Retrieved {len(df)} records")
                return df

        except Exception as e:
            self.log_manager.log_error("Data retrieval error", exc_info=True)
            raise


class LearningAnalyzer:
    """학습 과정 분석 클래스"""

    def __init__(self, log_manager):
        self.log_manager = log_manager
        self.learning_results = {}
        self.prize_counts = {}
        self.best_model = None
        self.best_score = 0

    def evaluate_model(self, analyzer, actual_numbers, iterations=100):
        """모델 성능 평가"""
        total_score = 0
        match_counts = {i: 0 for i in range(7)}  # 0~6개 일치 횟수

        for _ in range(iterations):
            predicted = analyzer.select_numbers()
            matches = len(set(predicted) & set(actual_numbers))
            total_score += self._calculate_match_score(matches)
            match_counts[matches] += 1

        average_score = total_score / iterations
        return average_score, match_counts

    def _calculate_match_score(self, matches):
        """일치 개수에 따른 점수 계산"""
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
        """회차별 당첨 번호 분석"""
        matches = set(predicted_numbers) & set(actual_numbers)
        match_count = len(matches)

        # 등수 판정
        prize_rank = self._get_prize_rank(match_count)

        # 회차별 통계 업데이트
        if draw_number not in self.prize_counts:
            self.prize_counts[draw_number] = {
                1: 0,  # 1등
                2: 0,  # 2등
                3: 0,  # 3등
                4: 0,  # 4등
                5: 0,  # 5등
                0: 0  # 미당첨
            }

        if prize_rank > 0:
            self.prize_counts[draw_number][prize_rank] += 1
        else:
            self.prize_counts[draw_number][0] += 1

        return {
            'draw_number': draw_number,
            'actual_numbers': actual_numbers,
            'predicted_numbers': predicted_numbers,
            'matches': matches,
            'match_count': match_count,
            'prize_rank': prize_rank
        }

    def _get_prize_rank(self, match_count):
        """당첨 등수 판정"""
        prize_ranks = {
            6: 1,  # 1등
            5: 2,  # 2등
            4: 3,  # 3등
            3: 4,  # 4등
            2: 5  # 5등
        }
        return prize_ranks.get(match_count, 0)

    def get_draw_summary(self, draw_number):
        """회차별 학습 결과 요약"""
        if draw_number in self.prize_counts:
            counts = self.prize_counts[draw_number]
            total_tries = sum(counts.values())

            return {
                'draw_number': draw_number,
                'total_tries': total_tries,
                'prize_counts': counts,
                'success_rate': {
                    rank: (count / total_tries * 100) if total_tries > 0 else 0
                    for rank, count in counts.items()
                }
            }
        return None


class LottoAnalyzer:
    """로또 번호 분석 및 예측 클래스"""

    def __init__(self, learning_rate: float, log_manager: LogManager):
        self.learning_rate = learning_rate
        self.log_manager = log_manager
        self.numbers_memory = {i: 1.0 for i in range(1, 46)}
        self.number_stats = {i: 0 for i in range(1, 46)}

    def get_state(self) -> ModelState:
        """현재 모델 상태 반환"""
        return ModelState(
            numbers_memory=dict(self.numbers_memory),
            number_stats=dict(self.number_stats)
        )

    def set_state(self, state: ModelState):
        """모델 상태 설정"""
        self.numbers_memory = copy.deepcopy(state.numbers_memory)
        self.number_stats = copy.deepcopy(state.number_stats)

    def analyze_historical_data(self, df: pd.DataFrame):
        """과거 데이터 분석"""
        self.log_manager.log_info("Starting historical data analysis")
        total_records = len(df)

        for _, row in df.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 7)]
            for num in numbers:
                self.number_stats[num] += 1

        # 통계 정보 로깅
        for num, count in self.number_stats.items():
            frequency = count / total_records * 100
            self.log_manager.log_debug(f"Number {num}: {count} occurrences ({frequency:.2f}%)")

    def select_numbers(self) -> list:
        """가중치 기반 번호 선택"""
        weights = list(self.numbers_memory.values())
        weight_sum = sum(weights)
        probabilities = [w / weight_sum for w in weights]

        selected = []
        while len(selected) < 6:
            num = np.random.choice(range(1, 46), p=probabilities)
            if num not in selected:
                selected.append(num)
                # 선택된 번호의 가중치 일시적 감소 (다양성 확보)
                probabilities[int(num) - 1] *= 0.5
                probabilities = [p / sum(probabilities) for p in probabilities]

        selected.sort()
        return selected

    def update_weights(self, numbers: list, score: int):
        """가중치 업데이트"""
        for num in numbers:
            old_weight = self.numbers_memory[num]
            # 점수에 따른 가중치 증가
            increase = score * self.learning_rate
            # 현재 통계 기반 보정
            frequency = self.number_stats.get(num, 0)
            max_frequency = max(self.number_stats.values())
            frequency_factor = frequency / max_frequency if max_frequency > 0 else 0

            # 최종 가중치 업데이트
            self.numbers_memory[num] += increase * (1 + frequency_factor)

            self.log_manager.log_debug(
                f"Updated weight for number {num}: {old_weight:.2f} -> {self.numbers_memory[num]:.2f}"
            )


class LottoPredictionGUI:
    """로또 예측 시스템 GUI 클래스"""

    def __init__(self, root):
        self.root = root
        self.root.title("로또 번호 예측 시스템 v3.0")
        self.root.geometry("1200x800")

        # 폰트 설정
        self._setup_fonts()

        # 파일 및 로그 매니저 초기화
        self.file_manager = FileManager()
        self.log_manager = LogManager(self.file_manager)

        # 데이터베이스 매니저 초기화
        self.db_manager = DatabaseManager('lotto.db', self.log_manager)

        # GUI 초기화
        self._setup_gui()
        self._setup_visualization()

        self.is_running = False
        self.best_model_state = None
        self.log_manager.log_info("Application started successfully")

    def _setup_gui(self):
        """GUI 구성요소 초기화"""
        # 노트북(탭) 생성
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)

        # 탭 프레임 생성
        self.main_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.main_tab, text="예측")
        self.notebook.add(self.analysis_tab, text="분석")
        self.notebook.add(self.stats_tab, text="통계")

        self._create_main_tab()
        self._create_analysis_tab()
        self._create_stats_tab()

    def _create_main_tab(self):
        """메인 예측 탭 구성"""
        # 설정 프레임
        settings_frame = ttk.LabelFrame(self.main_tab, text="예측 설정", padding=10)
        settings_frame.pack(fill='x', padx=5, pady=5)

        # 게임 수 설정
        ttk.Label(settings_frame, text="게임 수:").grid(row=0, column=0, padx=5)
        self.games_var = tk.StringVar(value="5")
        ttk.Entry(settings_frame, textvariable=self.games_var, width=10).grid(row=0, column=1)

        # 학습 회차 설정
        ttk.Label(settings_frame, text="학습 회차:").grid(row=0, column=2, padx=5)
        self.learning_draws_var = tk.StringVar(value="100")
        self.learning_draws_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.learning_draws_var,
            values=["전체", "10", "30", "50", "100", "200"]
        )
        self.learning_draws_combo.grid(row=0, column=3, padx=5)

        # 학습률 설정
        ttk.Label(settings_frame, text="학습률:").grid(row=0, column=4, padx=5)
        self.learning_rate_var = tk.StringVar(value="0.1")
        ttk.Entry(settings_frame, textvariable=self.learning_rate_var, width=10).grid(row=0, column=5)

        # 반복 학습 횟수 설정
        ttk.Label(settings_frame, text="학습 반복:").grid(row=0, column=6, padx=5)
        self.iterations_var = tk.StringVar(value="100")
        ttk.Entry(settings_frame, textvariable=self.iterations_var, width=10).grid(row=0, column=7)

        # 실행 버튼
        self.run_button = ttk.Button(settings_frame, text="예측 시작", command=self.run_prediction)
        self.run_button.grid(row=0, column=8, padx=10)

        # 로그 창
        log_frame = ttk.LabelFrame(self.main_tab, text="실행 로그", padding=10)
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.pack(fill='both', expand=True)

        # 결과 창
        results_frame = ttk.LabelFrame(self.main_tab, text="예측 결과", padding=10)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=10)
        self.results_text.pack(fill='both', expand=True)

        # 상태바
        self.status_var = tk.StringVar(value="준비")
        status_bar = ttk.Label(self.main_tab, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill='x', padx=5, pady=5)

        # 프로그레스바
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.main_tab,
            length=300,
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(fill='x', padx=5)

    def _create_analysis_tab(self):
        """분석 탭 구성"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.analysis_tab)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _create_stats_tab(self):
        """통계 탭 구성"""
        self.stats_text = scrolledtext.ScrolledText(self.stats_tab, height=30)
        self.stats_text.pack(fill='both', expand=True)

    def _prediction_thread(self):
        """예측 실행 스레드"""
        try:
            self.status_var.set("데이터 로딩 중...")
            self.log_manager.log_info("Starting prediction process")

            # 학습 회차 설정 적용
            draw_limit = None if self.learning_draws_var.get() == "전체" else int(self.learning_draws_var.get())
            historical_data = self.db_manager.get_historical_data(draw_limit)
            learning_analyzer = LearningAnalyzer(self.log_manager)
            iterations = int(self.iterations_var.get())

            # 학습 시작 정보 로깅
            data_range = "전체" if not draw_limit else f"최근 {draw_limit}회차"
            self.log_manager.log_info(f"학습 설정: {data_range}, 반복 횟수: {iterations}")
            self.log_text.insert(tk.END, f"\n학습 설정:\n- 데이터: {data_range}\n- 반복 횟수: {iterations}\n")

            overall_best_state = None
            overall_best_score = 0

            # 각 회차별 학습
            total_draws = len(historical_data)
            for idx, row in historical_data.iterrows():
                if not self.is_running:
                    break

                draw_number = row['draw_number']
                actual_numbers = [row[f'num{i}'] for i in range(1, 7)]

                self.status_var.set(f"{draw_number}회차 학습 중... ({idx + 1}/{total_draws})")

                # 학습 시작 로그
                self.log_manager.log_info(f"\n=== {draw_number}회차 학습 시작 ===")
                self.log_manager.log_info(f"실제 당첨번호: {actual_numbers}")
                self.log_text.insert(tk.END, f"\n\n=== {draw_number}회차 학습 ===")
                self.log_text.insert(tk.END, f"\n실제 당첨번호: {actual_numbers}")

                # 현재 회차에서의 최고 성능 모델
                best_state = None
                best_score = 0
                match_summary = {i: 0 for i in range(7)}  # 0~6개 일치 횟수

                # iterations 횟수만큼 학습 및 평가
                for iteration in range(iterations):
                    if not self.is_running:
                        break

                    # 번호 예측 및 평가
                    predicted_numbers = self.analyzer.select_numbers()
                    matches = len(set(predicted_numbers) & set(actual_numbers))
                    match_summary[matches] += 1

                    current_score, _ = learning_analyzer.evaluate_model(
                        self.analyzer, actual_numbers
                    )

                    if current_score > best_score:
                        best_score = current_score
                        best_state = self.analyzer.get_state()
                        best_state.score = current_score

                        if current_score > overall_best_score:
                            overall_best_score = current_score
                            overall_best_state = best_state.copy()

                    # 가중치 업데이트
                    if matches > 0:
                        self.analyzer.update_weights(
                            list(set(predicted_numbers) & set(actual_numbers)),
                            matches
                        )

                    # 진행 상황 표시
                    if iteration % 10 == 0:
                        progress = (idx * iterations + iteration) / (total_draws * iterations) * 100
                        self.progress_var.set(progress)
                        self.root.update_idletasks()

                # 회차별 학습 결과 요약
                result = learning_analyzer.analyze_match(
                    draw_number, actual_numbers, predicted_numbers)

                summary_text = (
                    f"학습 완료:\n"
                    f"- 최고 점수: {best_score:.2f}\n"
                    f"- 매칭 통계:\n"
                )

                for matches, count in match_summary.items():
                    if count > 0:
                        percentage = (count / iterations * 100)
                        summary_text += f"  {matches}개 일치: {count}회 ({percentage:.2f}%)\n"

                self.log_manager.log_info(summary_text)
                self.log_text.insert(tk.END, f"\n{draw_number}회차: {summary_text}")
                self.log_text.see(tk.END)

            if self.is_running and overall_best_state:
                # 최고 성능 모델 저장
                self._save_model(overall_best_state)

                # 최고 성능 모델로 최종 예측 수행
                self.analyzer.set_state(overall_best_state)
                games = int(self.games_var.get())

                self.status_var.set("최종 예측 생성 중...")
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END,
                                      f"최고 성능 모델 점수: {overall_best_score:.2f}\n"
                                      f"학습 데이터: {data_range}\n"
                                      f"학습 반복 횟수: {iterations}\n\n"
                                      )

                # 최종 예측 번호 생성
                final_predictions = []
                for game in range(games):
                    predicted_numbers = self.analyzer.select_numbers()
                    final_predictions.append(predicted_numbers)
                    self.results_text.insert(tk.END, f"게임 {game + 1}: {predicted_numbers}\n")

                # 결과 저장
                self._save_learning_results(learning_analyzer)
                self._save_final_predictions(final_predictions)
                self._update_analysis_graphs(historical_data)
                self._update_stats_tab(historical_data)

                self.status_var.set("완료")
                self.log_manager.log_info(
                    f"Prediction process completed successfully\n"
                    f"Best model score: {overall_best_score:.2f}\n"
                    f"Generated {games} predictions"
                )
                messagebox.showinfo("완료",
                                  f"예측이 완료되었습니다!\n"
                                  f"최고 성능 점수: {overall_best_score:.2f}"
                                  )

        except Exception as e:
            self.log_manager.log_error("Prediction process error", exc_info=True)
            self.status_var.set("오류 발생")
            messagebox.showerror("오류", f"예측 중 오류가 발생했습니다: {str(e)}")
        finally:
            self.is_running = False
            self.run_button.config(text="예측 시작")
            self.progress_var.set(0)

    def run_prediction(self):
        """예측 실행"""
        try:
            if self.is_running:
                self.is_running = False
                self.run_button.config(text="예측 시작")
                return

            # 입력값 검증
            games = int(self.games_var.get())
            learning_rate = float(self.learning_rate_var.get())
            iterations = int(self.iterations_var.get())

            if games > 50:
                raise ValueError("최대 50게임까지만 가능합니다")
            if learning_rate <= 0 or learning_rate > 1:
                raise ValueError("학습률은 0과 1 사이여야 합니다")
            if iterations <= 0:
                raise ValueError("학습 반복 횟수는 1 이상이어야 합니다")

            # 분석기 초기화
            self.analyzer = LottoAnalyzer(learning_rate, self.log_manager)

            # 스레드 시작
            self.is_running = True
            self.run_button.config(text="중지")
            thread = threading.Thread(target=self._prediction_thread)
            thread.daemon = True
            thread.start()

        except ValueError as e:
            self.log_manager.log_error(f"Validation error: {str(e)}")
            messagebox.showerror("입력 오류", str(e))
        except Exception as e:
            self.log_manager.log_error("Prediction initialization error", exc_info=True)
            messagebox.showerror("오류", f"예측 실행 중 오류가 발생했습니다: {str(e)}")

    def _setup_visualization(self):
        """시각화 초기 설정"""
        try:
            plt.style.use('default')

            # 그래프 설정
            self.fig.suptitle('로또 번호 분석')
            self.ax1.set_title('번호별 출현 빈도')
            self.ax2.set_title('당첨금 트렌드')

            # 그리드 설정
            self.ax1.grid(True, linestyle='--', alpha=0.7)
            self.ax2.grid(True, linestyle='--', alpha=0.7)

            # 축 레이블 설정
            self.ax1.set_xlabel('번호')
            self.ax1.set_ylabel('출현 횟수')
            self.ax2.set_xlabel('회차')
            self.ax2.set_ylabel('당첨금')

            # 여백 조정
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        except Exception as e:
            self.log_manager.log_error(f"Visualization setup error: {str(e)}", exc_info=True)
            raise

    def _update_analysis_graphs(self, historical_data: pd.DataFrame):
        """분석 그래프 업데이트"""
        try:
            self.ax1.clear()
            self.ax2.clear()

            # 번호별 출현 빈도 그래프
            numbers = list(range(1, 46))
            frequencies = [self.analyzer.number_stats[n] for n in numbers]
            self.ax1.bar(numbers, frequencies)
            self.ax1.set_title('번호별 출현 빈도')
            self.ax1.set_xlabel('번호')
            self.ax1.set_ylabel('출현 횟수')
            self.ax1.grid(True, linestyle='--', alpha=0.7)

            # 당첨금 트렌드 그래프
            draws = historical_data['draw_number']
            prizes = historical_data['money1']
            self.ax2.plot(draws, prizes)
            self.ax2.set_title('1등 당첨금 트렌드')
            self.ax2.set_xlabel('회차')
            self.ax2.set_ylabel('당첨금')
            self.ax2.grid(True, linestyle='--', alpha=0.7)

            # 그래프 업데이트
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.canvas.draw()

        except Exception as e:
            self.log_manager.log_error("Error updating analysis graphs", exc_info=True)
            raise

    def _save_model(self, model_state):
        """최고 성능 모델 저장"""
        try:
            model_path = self.file_manager.get_model_file()
            model_state.save_to_file(model_path)
            self.log_manager.log_info(f"Best model saved to: {model_path}")
        except Exception as e:
            self.log_manager.log_error("Error saving model", exc_info=True)

    def _analyze_historical_data(self, df: pd.DataFrame):
        """과거 데이터 분석"""
        number_stats = {i: 0 for i in range(1, 46)}
        total_draws = len(df)

        for _, row in df.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 7)]
            for num in numbers:
                number_stats[num] += 1

        return number_stats, total_draws

    def _update_stats_tab(self, historical_data: pd.DataFrame):
        """통계 탭 업데이트"""
        try:
            self.stats_text.delete(1.0, tk.END)

            # 번호별 통계 계산
            number_stats, total_draws = self._analyze_historical_data(historical_data)

            stats_text = f"전체 회차 수: {total_draws}\n\n"
            stats_text += "번호별 통계:\n"

            for num in range(1, 46):
                count = number_stats[num]
                frequency = (count / total_draws) * 100 if total_draws > 0 else 0
                stats_text += f"번호 {num:2d}: {count:4d}회 출현 ({frequency:5.2f}%)\n"

            self.stats_text.insert(tk.END, stats_text)

        except Exception as e:
            self.log_manager.log_error("Error updating statistics", exc_info=True)
            raise

    def _save_final_predictions(self, predictions):
        """최종 예측 번호 저장"""
        try:
            # CSV 저장
            csv_path = self.file_manager.get_prediction_file('csv')
            pd.DataFrame(predictions, columns=[f'번호{i + 1}' for i in range(6)]).to_csv(
                csv_path, index=False, encoding='utf-8-sig'
            )

            # Excel 저장
            excel_path = self.file_manager.get_prediction_file('xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                pd.DataFrame(predictions, columns=[f'번호{i + 1}' for i in range(6)]).to_excel(
                    writer, sheet_name='예측번호', index=False
                )

            self.log_manager.log_info(f"Predictions saved to {csv_path} and {excel_path}")
        except Exception as e:
            self.log_manager.log_error("Error saving predictions", exc_info=True)
            raise

    def _setup_fonts(self):
        """폰트 설정"""
        try:
            if sys.platform == 'win32':
                font_path = 'C:/Windows/Fonts/malgun.ttf'
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
            else:
                plt.rcParams['font.family'] = 'NanumGothic'

            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 10

            # GUI 폰트 설정
            if sys.platform == 'win32':
                default_font = ('맑은 고딕', 9)
            else:
                default_font = ('NanumGothic', 9)

            style = ttk.Style()
            style.configure('TLabel', font=default_font)
            style.configure('TButton', font=default_font)

            self.font_config = {
                'default': default_font,
                'header': (default_font[0], 11),
                'title': (default_font[0], 10)
            }

        except Exception as e:
            self.log_manager.log_error(f"Font setup error: {str(e)}", exc_info=True)
            # 폰트 설정 실패시 기본 폰트 사용
            plt.rcParams['font.family'] = 'sans-serif'

    def _save_learning_results(self, learning_analyzer):
        """학습 결과 저장"""
        try:
            excel_path = self.file_manager.get_prediction_file('xlsx')

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # 회차별 결과
                draw_results = []
                for draw_number in learning_analyzer.prize_counts.keys():
                    summary = learning_analyzer.get_draw_summary(draw_number)
                    if summary:
                        draw_results.append({
                            '회차': draw_number,
                            '총 시도횟수': summary['total_tries'],
                            '1등 당첨': summary['prize_counts'][1],
                            '2등 당첨': summary['prize_counts'][2],
                            '3등 당첨': summary['prize_counts'][3],
                            '4등 당첨': summary['prize_counts'][4],
                            '5등 당첨': summary['prize_counts'][5],
                            '미당첨': summary['prize_counts'][0],
                            '1등 확률': f"{summary['success_rate'][1]:.2f}%",
                            '2등 확률': f"{summary['success_rate'][2]:.2f}%",
                            '3등 확률': f"{summary['success_rate'][3]:.2f}%",
                            '4등 확률': f"{summary['success_rate'][4]:.2f}%",
                            '5등 확률': f"{summary['success_rate'][5]:.2f}%",
                            '미당첨 확률': f"{summary['success_rate'][0]:.2f}%"
                        })

                # 회차별 학습 결과 저장
                df_results = pd.DataFrame(draw_results)
                df_results.to_excel(writer, sheet_name='회차별_학습결과', index=False)

                # 학습 설정 정보 저장
                settings_data = {
                    '설정': ['학습률', '학습 반복 횟수', '게임 수', '학습 회차'],
                    '값': [
                        self.learning_rate_var.get(),
                        self.iterations_var.get(),
                        self.games_var.get(),
                        self.learning_draws_var.get()
                    ]
                }
                pd.DataFrame(settings_data).to_excel(writer, sheet_name='학습설정', index=False)

                # 번호별 통계 저장
                if hasattr(self, 'analyzer'):
                    stats_data = {
                        '번호': list(range(1, 46)),
                        '가중치': [self.analyzer.numbers_memory[i] for i in range(1, 46)],
                        '출현횟수': [self.analyzer.number_stats[i] for i in range(1, 46)]
                    }
                    pd.DataFrame(stats_data).to_excel(writer, sheet_name='번호별통계', index=False)

            self.log_manager.log_info(f"Learning results saved to: {excel_path}")

        except Exception as e:
            self.log_manager.log_error("Error saving learning results", exc_info=True)
            raise

def main():
    try:
        root = tk.Tk()
        app = LottoPredictionGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
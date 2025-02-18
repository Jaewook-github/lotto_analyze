import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
from datetime import datetime
import os
import threading
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import seaborn as sns
from typing import List, Dict, Tuple
import logging


class DatabaseManager:
    """데이터베이스 연결 및 쿼리 관리 클래스"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self._setup_logging()

    def _setup_logging(self):
        """로깅 설정"""
        # 로그 파일 핸들러 생성 (인코딩 설정 포함)
        log_filename = 'lotto_prediction.log'
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')

        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 로거 설정
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)


    def connect(self):
        """데이터베이스 연결"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            return self.connection
        except Error as e:
            logging.error(f"Database connection error: {str(e)}")
            raise

    def get_historical_data(self, limit: int = None) -> pd.DataFrame:
        """당첨 이력 조회"""
        try:
            query = """
                SELECT draw_number, num1, num2, num3, num4, num5, num6, bonus, 
                       money1, money2, money3, money4, money5
                FROM lotto_results
                ORDER BY draw_number DESC
            """
            if limit:
                query += f" LIMIT {limit}"

            with self.connect() as conn:
                df = pd.read_sql_query(query, conn)
                logging.info(f"Retrieved {len(df)} historical records")
                return df
        except Error as e:
            logging.error(f"Data retrieval error: {str(e)}")
            raise


class LottoAnalyzer:
    """로또 번호 분석 및 예측 클래스"""

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.numbers_memory = defaultdict(lambda: 1.0)
        self.number_stats = defaultdict(int)
        self.pattern_stats = defaultdict(int)

    def analyze_historical_data(self, df: pd.DataFrame):
        """과거 데이터 분석"""
        for _, row in df.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 7)]
            bonus = row['bonus']

            # 개별 번호 통계
            for num in numbers:
                self.number_stats[num] += 1
            self.number_stats[bonus] += 0.5  # 보너스 번호는 가중치 절반

            # 패턴 분석 (연속된 번호, 간격 등)
            self._analyze_patterns(numbers)

    def _analyze_patterns(self, numbers: List[int]):
        """번호 패턴 분석"""
        numbers.sort()

        # 연속된 번호 패턴
        for i in range(len(numbers) - 1):
            if numbers[i + 1] - numbers[i] == 1:
                self.pattern_stats['consecutive'] += 1

        # 번호 간격 패턴
        gaps = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        self.pattern_stats[f'avg_gap_{int(avg_gap)}'] += 1

    def select_numbers(self) -> List[int]:
        """가중치 기반 번호 선택"""
        weights = [self.numbers_memory[i] for i in range(1, 46)]
        weights = np.array(weights) / sum(weights)

        selected = []
        while len(selected) < 6:
            num = np.random.choice(range(1, 46), p=weights)
            if num not in selected:
                selected.append(num)
                # 선택된 번호의 가중치 일시적 감소 (다양성 확보)
                weights[num - 1] *= 0.5
                weights = weights / sum(weights)

        return sorted(selected)

    def update_weights(self, numbers: List[int], score: int):
        """가중치 업데이트"""
        for num in numbers:
            base_update = score * self.learning_rate

            # 통계 기반 보정
            frequency_factor = self.number_stats[num] / max(self.number_stats.values())
            pattern_factor = self._get_pattern_factor(num)

            total_update = base_update * (frequency_factor + pattern_factor) / 2
            self.numbers_memory[num] += total_update

    def _get_pattern_factor(self, num: int) -> float:
        """패턴 기반 가중치 보정 계수 계산"""
        pattern_count = sum(1 for pattern, count in self.pattern_stats.items()
                            if str(num) in pattern)
        return pattern_count / max(1, len(self.pattern_stats))


class LottoPredictionGUI:
    """로또 예측 시스템 GUI 클래스"""

    def __init__(self, root):
        self.root = root
        self.root.title("고급 로또 번호 예측 시스템 v2.0")
        self.root.geometry("1200x800")

        self.db_manager = DatabaseManager('lotto.db')
        self.analyzer = None  # 실행 시 초기화

        self._setup_gui()
        self._setup_visualization()

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

        # 설정 컨트롤
        ttk.Label(settings_frame, text="게임 수:").grid(row=0, column=0, padx=5)
        self.games_var = tk.StringVar(value="5")
        ttk.Entry(settings_frame, textvariable=self.games_var, width=10).grid(row=0, column=1)

        ttk.Label(settings_frame, text="학습 회차:").grid(row=0, column=2, padx=5)
        self.learning_draws_var = tk.StringVar(value="100")
        ttk.Entry(settings_frame, textvariable=self.learning_draws_var, width=10).grid(row=0, column=3)

        ttk.Label(settings_frame, text="학습률:").grid(row=0, column=4, padx=5)
        self.learning_rate_var = tk.StringVar(value="0.1")
        ttk.Entry(settings_frame, textvariable=self.learning_rate_var, width=10).grid(row=0, column=5)

        # 실행 버튼
        self.run_button = ttk.Button(settings_frame, text="예측 시작", command=self.run_prediction)
        self.run_button.grid(row=0, column=6, padx=10)

        # 로그 영역
        log_frame = ttk.LabelFrame(self.main_tab, text="실행 로그", padding=10)
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.pack(fill='both', expand=True)

        # 결과 영역
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
        self.progress_bar = ttk.Progressbar(self.main_tab, length=300, mode='determinate',
                                            variable=self.progress_var)
        self.progress_bar.pack(fill='x', padx=5)

    def _create_analysis_tab(self):
        """분석 탭 구성"""
        # 그래프 영역
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.analysis_tab)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _create_stats_tab(self):
        """통계 탭 구성"""
        stats_frame = ttk.Frame(self.stats_tab)
        stats_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # 통계 테이블
        columns = ('번호', '출현 횟수', '확률', '최근 출현')
        self.stats_tree = ttk.Treeview(stats_frame, columns=columns, show='headings')

        for col in columns:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=100)

        self.stats_tree.pack(fill='both', expand=True)

        # 스크롤바
        scrollbar = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_tree.yview)
        scrollbar.pack(side='right', fill='y')

        self.stats_tree.configure(yscrollcommand=scrollbar.set)

    def _setup_visualization(self):
        """시각화 초기 설정"""
        # seaborn 스타일 대신 기본 스타일 사용
        plt.style.use('default')

        # 한글 폰트 설정 (필요한 경우)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 그래프 설정
        self.fig.suptitle('로또 번호 분석', fontsize=12)
        self.ax1.set_title('번호별 출현 빈도', fontsize=10)
        self.ax2.set_title('당첨금 트렌드', fontsize=10)

        # 그래프 스타일 설정
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        self.ax2.grid(True, linestyle='--', alpha=0.7)

        # 여백 조정
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def run_prediction(self):
        """예측 실행"""
        try:
            # 입력값 검증
            games = int(self.games_var.get())
            learning_draws = int(self.learning_draws_var.get())
            learning_rate = float(self.learning_rate_var.get())

            if games > 50:
                raise ValueError("최대 50게임까지만 가능합니다")
            if learning_rate <= 0 or learning_rate > 1:
                raise ValueError("학습률은 0과 1 사이여야 합니다")

            # 분석기 초기화
            self.analyzer = LottoAnalyzer(learning_rate)

            # 스레드 실행
            self.is_running = True
            thread = threading.Thread(target=self._prediction_thread,
                                      args=(games, learning_draws))
            thread.daemon = True
            thread.start()

        except ValueError as e:
            messagebox.showerror("입력 오류", str(e))
        except Exception as e:
            messagebox.showerror("실행 오류", f"예측 실행 중 오류가 발생했습니다: {str(e)}")

    def _prediction_thread(self, games: int, learning_draws: int):
        """예측 실행 스레드"""
        try:
            self.status_var.set("데이터 로딩 중...")
            historical_data = self.db_manager.get_historical_data(learning_draws)

            self.status_var.set("데이터 분석 중...")
            self.analyzer.analyze_historical_data(historical_data)

            predictions = []
            self.log_text.delete(1.0, tk.END)
            self.results_text.delete(1.0, tk.END)

            for game in range(games):
                if not self.is_running:
                    break

                self.status_var.set(f"게임 {game + 1} 예측 중...")
                selected_numbers = self.analyzer.select_numbers()

                # 예측 번호와 과거 데이터 비교
                matches_log = []
                for _, row in historical_data.iterrows():
                    actual = [row[f'num{i}'] for i in range(1, 7)]
                    matches = set(selected_numbers) & set(actual)

                    if matches:
                        matches_log.append({
                            'draw_number': row['draw_number'],
                            'matches': matches,
                            'count': len(matches)
                        })
                        self.analyzer.update_weights(list(matches), len(matches))

                # 로그 기록
                self._log_prediction_details(game + 1, selected_numbers, matches_log)

                # 결과 저장
                predictions.append(selected_numbers)

                # 진행률 업데이트
                self.progress_var.set((game + 1) / games * 100)

            if self.is_running:
                self._save_results(predictions)
                self._update_analysis_graphs(historical_data)
                self._update_statistics(predictions, historical_data)

                self.status_var.set("완료")
                messagebox.showinfo("완료", "예측이 완료되었습니다!")

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            self.status_var.set("오류 발생")
            messagebox.showerror("오류", f"예측 중 오류가 발생했습니다: {str(e)}")
        finally:
            self.is_running = False
            self.progress_var.set(0)
            self.run_button.config(text="예측 시작")

    def _log_prediction_details(self, game: int, numbers: List[int], matches_log: List[Dict]):
        """예측 상세 정보 로깅"""
        log_text = f"\n=== 게임 {game} 예측 결과 ===\n"
        log_text += f"선택된 번호: {numbers}\n"

        if matches_log:
            log_text += "\n과거 당첨 번호와 비교:\n"
            for match in matches_log[:5]:  # 상위 5개만 표시
                log_text += f"- {match['draw_number']}회차: {match['count']}개 일치 {match['matches']}\n"

        self.log_text.insert(tk.END, log_text)
        self.log_text.see(tk.END)

        self.results_text.insert(tk.END, f"게임 {game}: {numbers}\n")
        self.results_text.see(tk.END)

    def _save_results(self, predictions: List[List[int]]):
        """예측 결과 저장"""
        today = datetime.now().strftime('%Y%m%d_%H%M%S')

        # DataFrame 생성
        df = pd.DataFrame(predictions, columns=[f'번호{i + 1}' for i in range(6)])

        # CSV 저장
        csv_filename = f'로또예측_{today}.csv'
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

        # Excel 저장
        excel_filename = f'로또예측_{today}.xlsx'
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='예측번호', index=False)

            # 분석 정보 시트 추가
            analysis_data = pd.DataFrame({
                '번호': list(self.analyzer.number_stats.keys()),
                '출현횟수': list(self.analyzer.number_stats.values())
            })
            analysis_data.to_excel(writer, sheet_name='분석정보', index=False)

            # 패턴 정보 시트 추가
            pattern_data = pd.DataFrame({
                '패턴': list(self.analyzer.pattern_stats.keys()),
                '발생횟수': list(self.analyzer.pattern_stats.values())
            })
            pattern_data.to_excel(writer, sheet_name='패턴분석', index=False)

    def _update_analysis_graphs(self, historical_data: pd.DataFrame):
        """분석 그래프 업데이트"""
        self.ax1.clear()
        self.ax2.clear()

        # 번호별 출현 빈도 그래프
        numbers = list(range(1, 46))
        frequencies = [self.analyzer.number_stats[n] for n in numbers]
        self.ax1.bar(numbers, frequencies)
        self.ax1.set_title('번호별 출현 빈도')
        self.ax1.set_xlabel('번호')
        self.ax1.set_ylabel('출현 횟수')

        # 당첨금 트렌드 그래프
        draws = historical_data['draw_number']
        prizes = historical_data['money1']
        self.ax2.plot(draws, prizes)
        self.ax2.set_title('1등 당첨금 트렌드')
        self.ax2.set_xlabel('회차')
        self.ax2.set_ylabel('당첨금')

        self.canvas.draw()

    def _update_statistics(self, predictions: List[List[int]], historical_data: pd.DataFrame):
        """통계 정보 업데이트"""
        self.stats_tree.delete(*self.stats_tree.get_children())

        for num in range(1, 46):
            freq = self.analyzer.number_stats[num]
            prob = freq / len(historical_data) * 100

            # 최근 출현 회차 찾기
            last_appearance = "없음"
            for _, row in historical_data.iterrows():
                if num in [row[f'num{i}'] for i in range(1, 7)]:
                    last_appearance = f"{row['draw_number']}회차"
                    break

            self.stats_tree.insert('', 'end', values=(
                num, freq, f"{prob:.2f}%", last_appearance
            ))

def main():
    root = tk.Tk()
    app = LottoPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
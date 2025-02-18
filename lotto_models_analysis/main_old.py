
import sys
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QTextEdit, QSpinBox, QProgressBar, QMessageBox,
                             QGroupBox, QTabWidget, QDoubleSpinBox, QFormLayout, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import logging
from utils.data_loader import DataLoader
from models import get_model


def setup_logging():
    """로깅 설정"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=f'logs/main_{datetime.now():%Y%m%d_%H%M%S}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )


class PredictionWorker(QThread):
    """예측 작업을 처리하는 워커 스레드"""

    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    log = pyqtSignal(str)

    def __init__(self, model_type, num_sets, db_path, start_draw=1, end_draw=None,
                 sequence_mode="전체 데이터", sequence_length=10):
        super().__init__()
        self.model_type = model_type
        self.num_sets = num_sets
        self.db_path = db_path
        self.start_draw = start_draw
        self.end_draw = end_draw
        self.sequence_mode = sequence_mode
        self.sequence_length = sequence_length
        self.data_loader = None

    def run(self):
        """예측 작업 실행"""
        try:
            # 데이터 로더 초기화
            self.data_loader = DataLoader(
                self.db_path,
                self.start_draw,
                self.end_draw,
                self.sequence_mode,
                self.sequence_length
            )

            self.log.emit("데이터 로드 중...")
            df = self.data_loader.load_data()

            if df.empty:
                raise Exception("데이터를 불러올 수 없습니다.")

            self.log.emit(f"총 {len(df):,}회차의 로또 데이터 로드 완료")

            # 데이터 전처리
            X, y, numbers = self.data_loader.preprocess_data(df)
            self.log.emit(f"\n데이터 전처리 결과:")
            self.log.emit(f"입력 데이터 shape: {X.shape}")
            self.log.emit(f"출력 데이터 shape: {y.shape}")

            predictions = []

            try:
                # 모델 생성
                model = get_model(self.model_type)
                self.log.emit(f"\n{self.model_type} 처리 시작...")

                # 학습 데이터 준비
                if self.model_type == "혼합 모델":
                    # numbers = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values[-len(X):]
                    X_cnn = self.data_loader.create_marking_patterns(numbers)
                    train_data = (X_cnn, X, y)
                else:
                    train_data = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values

                # 모델 학습
                self.log.emit("모델 학습 중...")
                model.train(train_data)

                # 번호 예측
                self.log.emit("\n번호 예측 중...")
                for i in range(self.num_sets):
                    try:
                        if self.model_type == "혼합 모델":
                            pred = model.predict_numbers((X_cnn, X))
                        else:
                            pred = model.predict_numbers(train_data)

                        if pred:
                            predictions.append(pred)
                            self.progress.emit(int((i + 1) / self.num_sets * 100))
                            self.log.emit(f"세트 {i + 1} 예측 완료: {pred}")
                        else:
                            raise ValueError("예측 결과가 없습니다.")

                    except Exception as e:
                        self.log.emit(f"세트 {i + 1} 예측 실패: {str(e)}")

                if not predictions:
                    raise Exception("예측 결과를 생성할 수 없습니다.")

                # 결과 저장
                self.save_results(predictions)

            except Exception as e:
                self.log.emit(f"모델 처리 중 오류 발생: {str(e)}")
                raise

            self.finished.emit(predictions)

        except Exception as e:
            error_msg = f"에러 발생: {str(e)}"
            self.log.emit(error_msg)
            logging.error(error_msg)
            self.finished.emit([])

    def save_results(self, predictions):
        """예측 결과 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('predictions', exist_ok=True)

            # CSV 파일로 저장
            with open(f'predictions/results_{timestamp}.csv', 'w',
                      encoding='utf-8') as f:
                f.write("세트,예측번호\n")
                for i, pred in enumerate(predictions, 1):
                    f.write(f"{i},{','.join(map(str, pred))}\n")

            self.log.emit(f"\n결과가 저장되었습니다: predictions/results_{timestamp}.csv")

        except Exception as e:
            self.log.emit(f"결과 저장 중 오류 발생: {str(e)}")


class LottoPredictionApp(QMainWindow):
    """로또 번호 예측 시스템 GUI"""

    def __init__(self):
        super().__init__()
        setup_logging()
        self.db_path = 'data/lotto.db'
        self.check_database()
        self.initUI()

    def check_database(self):
        """데이터베이스 연결 확인"""
        self.data_loader = DataLoader(self.db_path)
        try:
            df = self.data_loader.load_data()
            self.min_draw = df['draw_number'].min()
            self.max_draw = df['draw_number'].max()
            logging.info(f"데이터베이스 연결 성공: {len(df)}개의 로또 회차 데이터")
        except Exception as e:
            logging.error(f"데이터베이스 연결 실패: {str(e)}")
            self.show_error("데이터베이스 오류",
                            "로또 데이터베이스에 연결할 수 없습니다.")
            sys.exit(1)

    def initUI(self):
        """GUI 초기화"""
        self.setWindowTitle('로또 번호 예측 시스템')
        self.setGeometry(100, 100, 800, 600)

        # 메인 위젯과 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 탭 위젯 생성
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # 탭 추가
        tab_widget.addTab(self.create_data_tab(), "데이터 설정")
        tab_widget.addTab(self.create_model_tab(), "모델 설정")
        tab_widget.addTab(self.create_help_tab(), "도움말")

        # 데이터 설정 그룹
        data_group = QGroupBox("데이터 설정")
        data_layout = QVBoxLayout()

        # 회차 범위 선택
        draw_range_layout = QHBoxLayout()
        self.start_draw = QSpinBox()
        self.start_draw.setRange(self.min_draw, self.max_draw)
        self.start_draw.setValue(self.min_draw)
        self.start_draw.valueChanged.connect(self.validate_settings)

        self.end_draw = QSpinBox()
        self.end_draw.setRange(self.min_draw, self.max_draw)
        self.end_draw.setValue(self.max_draw)
        self.end_draw.valueChanged.connect(self.validate_settings)

        draw_range_layout.addWidget(QLabel("시작 회차:"))
        draw_range_layout.addWidget(self.start_draw)
        draw_range_layout.addWidget(QLabel("종료 회차:"))
        draw_range_layout.addWidget(self.end_draw)

        data_layout.addLayout(draw_range_layout)

        # 시퀀스 설정
        sequence_layout = QHBoxLayout()
        self.sequence_mode = QComboBox()
        self.sequence_mode.addItems(["전체 데이터", "지정 길이"])
        self.sequence_mode.currentTextChanged.connect(self.toggle_sequence_length)

        self.sequence_length = QSpinBox()
        self.sequence_length.setRange(5, 50)
        self.sequence_length.setValue(10)
        self.sequence_length.setEnabled(False)

        sequence_layout.addWidget(QLabel("시퀀스 모드:"))
        sequence_layout.addWidget(self.sequence_mode)
        sequence_layout.addWidget(QLabel("시퀀스 길이:"))
        sequence_layout.addWidget(self.sequence_length)

        data_layout.addLayout(sequence_layout)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # 모델 설정 그룹
        model_group = QGroupBox("모델 설정")
        model_layout = QHBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "혼합 모델",
            "강화학습 모델",
            "유전 알고리즘 모델",
            "트랜스포머 모델",
            "통계 기반 모델",
            "앙상블 모델"
        ])

        self.sets_spin = QSpinBox()
        self.sets_spin.setRange(1, 20)

        model_layout.addWidget(QLabel("모델 선택:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(QLabel("게임 세트:"))
        model_layout.addWidget(self.sets_spin)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 실행 버튼
        self.predict_btn = QPushButton("예측 시작")
        self.predict_btn.clicked.connect(self.start_prediction)
        layout.addWidget(self.predict_btn)

        # 진행 상황 표시
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # 로그 창
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel("실행 로그:"))
        layout.addWidget(self.log_text)

        # 결과 창
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(QLabel("예측 결과:"))
        layout.addWidget(self.result_text)

    def toggle_sequence_length(self, mode):
        """시퀀스 모드에 따른 길이 설정 활성화/비활성화"""
        self.sequence_length.setEnabled(mode == "지정 길이")
        self.validate_settings()

    def validate_settings(self):
        """설정 유효성 검사"""
        start = self.start_draw.value()
        end = self.end_draw.value()

        if start > end:
            self.end_draw.setValue(start)
            end = start

        data_length = end - start + 1
        min_required = (self.sequence_length.value() + 1
                        if self.sequence_mode.currentText() == "지정 길이"
                        else 11)

        if data_length < min_required:
            self.predict_btn.setEnabled(False)
            self.show_warning(
                "데이터 범위 경고",
                f"최소 {min_required}회차 이상의 데이터가 필요합니다."
            )
        else:
            self.predict_btn.setEnabled(True)

    def show_warning(self, title, message):
        """경고 메시지 표시"""
        QMessageBox.warning(self, title, message)

    def show_error(self, title, message):
        """에러 메시지 표시"""
        QMessageBox.critical(self, title, message)

    def start_prediction(self):
        """예측 작업 시작"""
        self.predict_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.result_text.clear()

        self.worker = PredictionWorker(
            self.model_combo.currentText(),
            self.sets_spin.value(),
            self.db_path,
            self.start_draw.value(),
            self.end_draw.value(),
            self.sequence_mode.currentText(),
            self.sequence_length.value()
        )

        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.show_results)
        self.worker.log.connect(self.update_log)

        self.log_text.append("예측 시작...")
        logging.info("예측 작업 시작")

        self.worker.start()

    def update_progress(self, value):
        """진행률 업데이트"""
        self.progress_bar.setValue(value)

    def update_log(self, message):
        """로그 메시지 업데이트"""
        self.log_text.append(message)
        logging.info(message)

        # main_old.py (계속)

    def show_results(self, predictions):
        """예측 결과 표시"""
        if not predictions:
            self.log_text.append("예측 실패!")
            self.predict_btn.setEnabled(True)
            return

        self.result_text.clear()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, pred in enumerate(predictions, 1):
            result = f"세트 {i:2d}: {', '.join(f'{num:2d}' for num in pred)}\n"
            self.result_text.append(result)
            logging.info(f"예측 결과 - 세트 {i:2d}: {pred}")

        self.predict_btn.setEnabled(True)
        self.log_text.append("\n예측 작업이 완료되었습니다!")
        self.log_text.append(f"\n결과가 저장된 위치:")
        self.log_text.append(f"predictions/results_{timestamp}.csv")
        logging.info("예측 작업 완료")

def main():
    """메인 함수"""
    try:
        # 필요한 디렉토리 생성
        for directory in ['logs', 'best_models', 'predictions', 'data']:
            os.makedirs(directory, exist_ok=True)

        # 애플리케이션 실행
        app = QApplication(sys.argv)
        ex = LottoPredictionApp()
        ex.show()
        sys.exit(app.exec_())

    except Exception as e:
        logging.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
import sys
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QComboBox,
                            QTextEdit, QSpinBox, QDoubleSpinBox, QProgressBar,
                            QMessageBox, QGroupBox, QFormLayout, QScrollArea,
                            QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import logging
import numpy as np
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
        self.setGeometry(100, 100, 1000, 800)

        # 메인 위젯과 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 탭 위젯 추가
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # 탭 생성
        self.main_tab = QWidget()
        self.settings_tab = QWidget()
        self.help_tab = QWidget()

        # 탭 추가
        self.tab_widget.addTab(self.main_tab, "메인")
        self.tab_widget.addTab(self.settings_tab, "설정")
        self.tab_widget.addTab(self.help_tab, "도움말")

        # 각 탭 초기화
        self.init_main_tab()
        self.init_settings_tab()
        self.init_help_tab()

    def init_main_tab(self):
        """메인 탭 초기화"""
        layout = QVBoxLayout(self.main_tab)

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

    def init_settings_tab(self):
        """설정 탭 초기화"""
        layout = QVBoxLayout(self.settings_tab)
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 하이브리드 모델 설정
        hybrid_group = QGroupBox("하이브리드 모델 (CNN + LSTM)")
        hybrid_layout = QFormLayout()


        self.hybrid_embedding = QSpinBox()
        self.hybrid_embedding.setRange(32, 128)
        self.hybrid_embedding.setValue(64)
        self.hybrid_dropout = QDoubleSpinBox()
        self.hybrid_dropout.setRange(0.1, 0.5)
        self.hybrid_dropout.setValue(0.3)
        self.hybrid_lr = QDoubleSpinBox()
        self.hybrid_lr.setRange(0.0001, 0.01)
        self.hybrid_lr.setValue(0.001)
        self.hybrid_lr.setSingleStep(0.0001)

        hybrid_layout.addRow("임베딩 차원 (32-128):", self.hybrid_embedding)
        hybrid_layout.addRow("드롭아웃 비율 (0.1-0.5):", self.hybrid_dropout)
        hybrid_layout.addRow("학습률 (0.0001-0.01):", self.hybrid_lr)
        hybrid_group.setLayout(hybrid_layout)
        scroll_layout.addWidget(hybrid_group)

        # 강화학습 모델 설정
        rl_group = QGroupBox("강화학습 모델 (DQN)")
        rl_layout = QFormLayout()

        self.rl_gamma = QDoubleSpinBox()
        self.rl_gamma.setRange(0.8, 0.99)
        self.rl_gamma.setValue(0.95)
        self.rl_epsilon = QDoubleSpinBox()
        self.rl_epsilon.setRange(0.1, 1.0)
        self.rl_epsilon.setValue(1.0)
        self.rl_memory = QSpinBox()
        self.rl_memory.setRange(1000, 5000)
        self.rl_memory.setValue(2000)

        rl_layout.addRow("감마 (할인율) (0.8-0.99):", self.rl_gamma)
        rl_layout.addRow("입실론 (탐험률) (0.1-1.0):", self.rl_epsilon)
        rl_layout.addRow("메모리 크기 (1000-5000):", self.rl_memory)
        rl_group.setLayout(rl_layout)
        scroll_layout.addWidget(rl_group)

        # 유전 알고리즘 모델 설정
        ga_group = QGroupBox("유전 알고리즘 모델")
        ga_layout = QFormLayout()

        self.ga_population = QSpinBox()
        self.ga_population.setRange(50, 200)
        self.ga_population.setValue(100)
        self.ga_mutation = QDoubleSpinBox()
        self.ga_mutation.setRange(0.05, 0.3)
        self.ga_mutation.setValue(0.1)
        self.ga_elite = QSpinBox()
        self.ga_elite.setRange(2, 10)
        self.ga_elite.setValue(4)

        ga_layout.addRow("개체군 크기 (50-200):", self.ga_population)
        ga_layout.addRow("돌연변이 비율 (0.05-0.3):", self.ga_mutation)
        ga_layout.addRow("엘리트 크기 (2-10):", self.ga_elite)
        ga_group.setLayout(ga_layout)
        scroll_layout.addWidget(ga_group)

        # 트랜스포머 모델 설정
        transformer_group = QGroupBox("트랜스포머 모델")
        transformer_layout = QFormLayout()

        self.trans_heads = QSpinBox()
        self.trans_heads.setRange(4, 16)
        self.trans_heads.setValue(8)
        self.trans_layers = QSpinBox()
        self.trans_layers.setRange(2, 8)
        self.trans_layers.setValue(4)
        self.trans_dim = QSpinBox()
        self.trans_dim.setRange(32, 256)
        self.trans_dim.setValue(64)

        transformer_layout.addRow("어텐션 헤드 수 (4-16):", self.trans_heads)
        transformer_layout.addRow("트랜스포머 층 수 (2-8):", self.trans_layers)
        transformer_layout.addRow("모델 차원 (32-256):", self.trans_dim)
        transformer_group.setLayout(transformer_layout)
        scroll_layout.addWidget(transformer_group)

        # 통계 기반 모델 설정
        stat_group = QGroupBox("통계 기반 모델")
        stat_layout = QFormLayout()

        self.stat_window = QSpinBox()
        self.stat_window.setRange(10, 100)
        self.stat_window.setValue(50)
        self.stat_weight_recent = QDoubleSpinBox()
        self.stat_weight_recent.setRange(0.1, 0.9)
        self.stat_weight_recent.setValue(0.4)

        stat_layout.addRow("분석 윈도우 크기 (10-100):", self.stat_window)
        stat_layout.addRow("최근 데이터 가중치 (0.1-0.9):", self.stat_weight_recent)
        stat_group.setLayout(stat_layout)
        scroll_layout.addWidget(stat_group)

        # 앙상블 모델 설정
        ensemble_group = QGroupBox("앙상블 모델")
        ensemble_layout = QFormLayout()

        self.ensemble_hybrid = QDoubleSpinBox()
        self.ensemble_hybrid.setRange(0.0, 1.0)
        self.ensemble_hybrid.setValue(0.3)
        self.ensemble_rl = QDoubleSpinBox()
        self.ensemble_rl.setRange(0.0, 1.0)
        self.ensemble_rl.setValue(0.2)
        self.ensemble_ga = QDoubleSpinBox()
        self.ensemble_ga.setRange(0.0, 1.0)
        self.ensemble_ga.setValue(0.2)
        self.ensemble_transformer = QDoubleSpinBox()
        self.ensemble_transformer.setRange(0.0, 1.0)
        self.ensemble_transformer.setValue(0.15)
        self.ensemble_stat = QDoubleSpinBox()
        self.ensemble_stat.setRange(0.0, 1.0)
        self.ensemble_stat.setValue(0.15)

        ensemble_layout.addRow("하이브리드 모델 가중치:", self.ensemble_hybrid)
        ensemble_layout.addRow("강화학습 모델 가중치:", self.ensemble_rl)
        ensemble_layout.addRow("유전 알고리즘 가중치:", self.ensemble_ga)
        ensemble_layout.addRow("트랜스포머 가중치:", self.ensemble_transformer)
        ensemble_layout.addRow("통계 모델 가중치:", self.ensemble_stat)
        ensemble_group.setLayout(ensemble_layout)
        scroll_layout.addWidget(ensemble_group)

        # 설정 버튼들
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("설정 저장")
        load_btn = QPushButton("설정 불러오기")
        default_btn = QPushButton("기본값 복원")

        save_btn.clicked.connect(self.save_settings)
        load_btn.clicked.connect(self.load_settings)
        default_btn.clicked.connect(self.restore_default_settings)

        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(default_btn)
        scroll_layout.addLayout(btn_layout)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

    def init_help_tab(self):
        """도움말 탭 초기화"""
        layout = QVBoxLayout(self.help_tab)
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 시스템 개요
        overview_group = QGroupBox("시스템 개요")
        overview_layout = QVBoxLayout()
        overview_text = QLabel(
            "이 시스템은 다양한 머신러닝 기법을 활용하여 로또 번호를 예측합니다. "
            "각각의 모델이 가진 장점들을 활용하여 더 안정적인 예측을 시도합니다.\n\n"
            "주요 특징:\n"
            "- 6가지 다른 접근 방식 활용\n"
            "- 실시간 데이터 업데이트\n"
            "- 다양한 설정 커스터마이징\n"
            "- 앙상블 기법을 통한 예측 안정화"
        )
        overview_text.setWordWrap(True)
        overview_layout.addWidget(overview_text)
        overview_group.setLayout(overview_layout)
        scroll_layout.addWidget(overview_group)

        # 모델별 설명
        models_group = QGroupBox("예측 모델 설명")
        models_layout = QVBoxLayout()
        models_text = QLabel(
            "1. 하이브리드 모델 (CNN + LSTM)\n"
            "   - CNN으로 마킹패턴 분석\n"
            "   - LSTM으로 시계열 패턴 학습\n"
            "   - 두 모델의 장점을 결합한 하이브리드 구조\n\n"
            "2. 강화학습 모델 (DQN)\n"
            "   - 순차적인 번호 선택 학습\n"
            "   - 보상 체계를 통한 최적화\n"
            "   - 경험 재생을 통한 학습 안정화\n\n"
            "3. 유전 알고리즘 모델\n"
            "   - 진화 알고리즘 기반 최적화\n"
            "   - 적합도 함수로 번호 조합 평가\n"
            "   - 세대를 거듭하며 개선\n\n"
            "4. 트랜스포머 모델\n"
            "   - 자기 주의 메커니즘 활용\n"
            "   - 복잡한 패턴 인식 가능\n"
            "   - 긴 시퀀스 처리에 효과적\n\n"
            "5. 통계 기반 모델\n"
            "   - 과거 데이터 통계 분석\n"
            "   - 빈도, 패턴, 확률 계산\n"
            "   - 데이터 기반 예측\n\n"
            "6. 앙상블 모델\n"
            "   - 여러 모델의 예측 통합\n"
            "   - 가중치 기반 투표 시스템\n"
            "   - 예측 안정성 향상"
        )
        models_text.setWordWrap(True)
        models_layout.addWidget(models_text)
        models_group.setLayout(models_layout)
        scroll_layout.addWidget(models_group)

        # 설정값 설명
        settings_group = QGroupBox("모델 설정값 설명")
        settings_layout = QVBoxLayout()
        settings_text = QLabel(
            "1. 하이브리드 모델 설정\n"
            "   - 임베딩 차원 (32-128): 데이터 표현의 복잡도. 높을수록 세밀한 패턴 학습 가능\n"
            "   - 드롭아웃 비율 (0.1-0.5): 과적합 방지 비율. 높을수록 일반화 능력 향상\n"
            "   - 학습률 (0.0001-0.01): 학습 속도 조절. 작을수록 안정적이나 학습 속도 감소\n\n"
            "2. 강화학습 모델 설정\n"
            "   - 감마 (0.8-0.99): 미래 보상의 중요도. 높을수록 장기적 이익 중시\n"
            "   - 입실론 (0.1-1.0): 탐험 비율. 높을수록 새로운 시도 증가\n"
            "   - 메모리 크기 (1000-5000): 학습 데이터 저장량. 클수록 다양한 경험 활용\n\n"
            "3. 유전 알고리즘 설정\n"
            "   - 개체군 크기 (50-200): 동시 평가할 번호 조합 수. 클수록 다양한 조합 탐색\n"
            "   - 돌연변이 비율 (0.05-0.3): 무작위 변화 확률. 높을수록 다양성 증가\n"
            "   - 엘리트 크기 (2-10): 보존할 우수 개체 수. 많을수록 안정성 증가\n\n"
            "4. 트랜스포머 모델 설정\n"
            "   - 어텐션 헤드 수 (4-16): 동시 분석할 패턴 수. 많을수록 복잡한 패턴 포착\n"
            "   - 트랜스포머 층 수 (2-8): 모델의 깊이. 깊을수록 복잡한 관계 학습\n"
            "   - 모델 차원 (32-256): 내부 표현의 크기. 클수록 더 많은 정보 처리\n\n"
            "5. 통계 기반 모델 설정\n"
            "   - 분석 윈도우 크기 (10-100): 참고할 과거 데이터 범위\n"
            "   - 최근 데이터 가중치 (0.1-0.9): 최근 데이터의 중요도\n\n"
            "6. 앙상블 모델 설정\n"
            "   - 각 모델별 가중치 (0.0-1.0): 예측 결과 반영 비율\n"
            "   - 가중치 합이 1이 되도록 자동 조정됨"
        )
        settings_text.setWordWrap(True)
        settings_layout.addWidget(settings_text)
        settings_group.setLayout(settings_layout)
        scroll_layout.addWidget(settings_group)

        # 사용 팁
        tips_group = QGroupBox("사용 팁")
        tips_layout = QVBoxLayout()
        tips_text = QLabel(
            "효과적인 사용을 위한 팁:\n\n"
            "1. 데이터 설정\n"
            "   - 충분한 양의 과거 데이터 사용 (최소 100회차 이상 권장)\n"
            "   - 시퀀스 길이는 분석하고자 하는 패턴의 주기 고려\n\n"
            "2. 모델 설정\n"
            "   - 처음에는 기본값으로 시작\n"
            "   - 한 번에 하나의 설정만 변경하여 영향 파악\n"
            "   - 성능이 좋은 설정은 따로 저장\n\n"
            "3. 결과 해석\n"
            "   - 여러 번의 예측 결과 비교\n"
            "   - 극단적인 설정은 피하기\n"
            "   - 앙상블 가중치는 각 모델의 성능에 따라 조정"
        )
        tips_text.setWordWrap(True)
        tips_layout.addWidget(tips_text)
        tips_group.setLayout(tips_layout)
        scroll_layout.addWidget(tips_group)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

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

    def save_settings(self):
        """설정값 저장"""
        try:
            settings = {
                'hybrid': {
                    'embedding': self.hybrid_embedding.value(),
                    'dropout': self.hybrid_dropout.value(),
                    'learning_rate': self.hybrid_lr.value()
                },
                'reinforcement': {
                    'gamma': self.rl_gamma.value(),
                    'epsilon': self.rl_epsilon.value(),
                    'memory': self.rl_memory.value()
                },
                'genetic': {
                    'population': self.ga_population.value(),
                    'mutation': self.ga_mutation.value(),
                    'elite': self.ga_elite.value()
                },
                'transformer': {
                    'heads': self.trans_heads.value(),
                    'layers': self.trans_layers.value(),
                    'dimension': self.trans_dim.value()
                },
                'statistical': {
                    'window': self.stat_window.value(),
                    'recent_weight': self.stat_weight_recent.value()
                },
                'ensemble': {
                    'hybrid': self.ensemble_hybrid.value(),
                    'reinforcement': self.ensemble_rl.value(),
                    'genetic': self.ensemble_ga.value(),
                    'transformer': self.ensemble_transformer.value(),
                    'statistical': self.ensemble_stat.value()
                }
            }

            np.save('settings.npy', settings)
            QMessageBox.information(self, "알림", "설정이 저장되었습니다.")

        except Exception as e:
            QMessageBox.warning(self, "경고", f"설정 저장 중 오류 발생: {str(e)}")

    def load_settings(self):
        """설정값 불러오기"""
        try:
            if os.path.exists('settings.npy'):
                settings = np.load('settings.npy', allow_pickle=True).item()

                # 각 모델별 설정 로드
                hybrid = settings['hybrid']
                self.hybrid_embedding.setValue(hybrid['embedding'])
                self.hybrid_dropout.setValue(hybrid['dropout'])
                self.hybrid_lr.setValue(hybrid['learning_rate'])

                rl = settings['reinforcement']
                self.rl_gamma.setValue(rl['gamma'])
                self.rl_epsilon.setValue(rl['epsilon'])
                self.rl_memory.setValue(rl['memory'])

                genetic = settings['genetic']
                self.ga_population.setValue(genetic['population'])
                self.ga_mutation.setValue(genetic['mutation'])
                self.ga_elite.setValue(genetic['elite'])

                transformer = settings['transformer']
                self.trans_heads.setValue(transformer['heads'])
                self.trans_layers.setValue(transformer['layers'])
                self.trans_dim.setValue(transformer['dimension'])

                statistical = settings['statistical']
                self.stat_window.setValue(statistical['window'])
                self.stat_weight_recent.setValue(statistical['recent_weight'])

                ensemble = settings['ensemble']
                self.ensemble_hybrid.setValue(ensemble['hybrid'])
                self.ensemble_rl.setValue(ensemble['reinforcement'])
                self.ensemble_ga.setValue(ensemble['genetic'])
                self.ensemble_transformer.setValue(ensemble['transformer'])
                self.ensemble_stat.setValue(ensemble['statistical'])

                QMessageBox.information(self, "알림", "설정을 불러왔습니다.")
            else:
                QMessageBox.warning(self, "경고", "저장된 설정이 없습니다.")

        except Exception as e:
            QMessageBox.warning(self, "경고", f"설정 불러오기 중 오류 발생: {str(e)}")

    def restore_default_settings(self):
        """기본 설정값 복원"""
        try:
            # 하이브리드 모델 기본값
            self.hybrid_embedding.setValue(64)
            self.hybrid_dropout.setValue(0.3)
            self.hybrid_lr.setValue(0.001)

            # 강화학습 모델 기본값
            self.rl_gamma.setValue(0.95)
            self.rl_epsilon.setValue(1.0)
            self.rl_memory.setValue(2000)

            # 유전 알고리즘 기본값
            self.ga_population.setValue(100)
            self.ga_mutation.setValue(0.1)
            self.ga_elite.setValue(4)

            # 트랜스포머 모델 기본값
            self.trans_heads.setValue(8)
            self.trans_layers.setValue(4)
            self.trans_dim.setValue(64)

            # 통계 모델 기본값
            self.stat_window.setValue(50)
            self.stat_weight_recent.setValue(0.4)

            # 앙상블 모델 기본값
            self.ensemble_hybrid.setValue(0.3)
            self.ensemble_rl.setValue(0.2)
            self.ensemble_ga.setValue(0.2)
            self.ensemble_transformer.setValue(0.15)
            self.ensemble_stat.setValue(0.15)

            QMessageBox.information(self, "알림", "기본 설정값이 복원되었습니다.")

        except Exception as e:
            QMessageBox.warning(self, "경고", f"기본값 복원 중 오류 발생: {str(e)}")

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
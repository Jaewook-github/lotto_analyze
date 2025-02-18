## *프로젝트 구조 개요:*
```
Copylotto_models_analysis/
├── main.py                 # 메인 GUI 애플리케이션
├── README.md
├── best_models/           # 학습된 모델 저장소
├── data/                  # 데이터 관리
├── logs/                  # 로그 파일
├── models/                # 예측 모델 구현
├── predictions/           # 예측 결과 저장
└── utils/                 # 유틸리티 기능
```

## *데이터 관리 시스템 (data/):*

### Lotto_Scraping_DB.py:
```
- 동행복권 웹사이트 스크래핑
- SQLite DB 관리 (lotto.db)
- 자동 데이터 업데이트
- 회차별 당첨번호 및 당첨금액 저장
```

## *유틸리티 모듈 (utils/):*

### a. data_loader.py:
```
주요 기능:
- DB 연결 및 데이터 로드
- 데이터 전처리
- 시퀀스 데이터 생성
- 마킹 패턴 생성
- 데이터 검증

핵심 메소드:
- load_data(): DB에서 데이터 로드
- preprocess_data(): 데이터 전처리
- create_marking_patterns(): 마킹 패턴 생성
- validate_data_range(): 데이터 범위 검증
```

### b. preprocessing.py:
```
주요 기능:
- 데이터 정규화
- 특징 추출
- 패턴 분석
- 시계열 처리

핵심 메소드:
- create_sequences(): 시계열 시퀀스 생성
- encode_numbers(): 번호 원-핫 인코딩
- calculate_number_stats(): 통계 계산
- analyze_patterns(): 패턴 분석
```

## *예측 모델 (models/):*

### a. base.py (BaseModel):
```
기본 기능:
- 모델 저장/로드
- 예측 검증
- 성능 평가
- 모델 정보 관리
```

### b. hybrid.py (HybridModel):
```
구조:
- CNN + LSTM 결합
- 배치 정규화
- 드롭아웃 적용
- 다중 출력층
```

### c. reinforcement.py (ReinforcementLearningModel):
```
특징:
- DQN 기반 학습
- 경험 재생
- ε-greedy 전략
- 타겟 네트워크
```

### d. genetic.py (GeneticAlgorithmModel):
```
구성:
- 개체군 관리
- 적합도 평가
- 교차/변이 연산
- 엘리트 보존
```

### e. statistical.py (StatisticalModel):
```
분석 요소:
- 빈도 분석
- 패턴 인식
- 확률 계산
- 트렌드 분석
```

### f. transformer.py (TransformerModel):
```
구조:
- 멀티헤드 어텐션
- 포지셔널 인코딩
- 피드포워드 네트워크
- 레이어 정규화
```

### g. ensemble.py (EnsembleModel):
```
특징:
- 다중 모델 통합
- 가중치 기반 투표
- 성능 기반 가중치 조정
- 예측 결과 통합
```

### 메인 애플리케이션 (main.py):
```
기능:
- GUI 인터페이스
- 실시간 예측
- 진행 상황 표시
- 결과 시각화
- 로그 관리
```
## *이 시스템의 주요 특징:*

### 다양한 접근 방식:
```
딥러닝 (Hybrid, Transformer)
강화학습 (DQN)
유전 알고리즘
통계 분석
앙상블 방식
```
### 데이터 관리:
```
자동 데이터 수집
실시간 업데이트
데이터 검증
효율적인 저장
```
### 모델 관리:
```
모델 저장/복원
성능 모니터링
하이퍼파라미터 관리
결과 검증
```
### 사용자 인터페이스:
```
직관적인 GUI
실시간 피드백
상세한 로깅
결과 저장
```


이 시스템은 복잡한 로또 번호 예측 작업을 다양한 알고리즘과 
방법론을 통해 접근하며, 사용자 친화적인 인터페이스를 제공합니다. 
각 모듈은 독립적으로 작동하면서도 상호 연계되어 있어 확장성과 
유지보수성이 뛰어납니다.
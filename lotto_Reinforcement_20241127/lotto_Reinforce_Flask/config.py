# config.py
import os
from pathlib import Path


class Config:
    # 기본 설정
    BASE_DIR = Path(__file__).resolve().parent
    STATIC_DIR = BASE_DIR / 'static'
    TEMPLATES_DIR = BASE_DIR / 'templates'

    # 데이터 저장 경로
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = DATA_DIR / 'logs'
    PREDICTIONS_DIR = DATA_DIR / 'predictions'
    MODELS_DIR = DATA_DIR / 'models'

    # 데이터베이스 설정
    DATABASE_PATH = DATA_DIR / 'lotto.db'

    # 웹 서버 설정
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('FLASK_ENV') == 'development'

    @classmethod
    def init_app(cls, app):
        # 필요한 디렉토리 생성
        for directory in [cls.DATA_DIR, cls.LOGS_DIR, cls.PREDICTIONS_DIR, cls.MODELS_DIR]:
            directory.mkdir(exist_ok=True)

        # 앱 설정 적용
        app.config.from_object(cls)



from .base import BaseModel
from .hybrid import HybridModel
from .reinforcement import ReinforcementLearningModel
from .genetic import GeneticAlgorithmModel
from .transformer import TransformerModel
from .statistical import StatisticalModel
from .ensemble import EnsembleModel

__all__ = [
    'BaseModel',
    'HybridModel',
    'ReinforcementLearningModel',
    'GeneticAlgorithmModel',
    'TransformerModel',
    'StatisticalModel',
    'EnsembleModel',
    'get_model'
]

def get_available_models():
    """사용 가능한 모델 목록 반환"""
    return {
        "혼합 모델": HybridModel,
        "강화학습 모델": ReinforcementLearningModel,
        "유전 알고리즘 모델": GeneticAlgorithmModel,
        "트랜스포머 모델": TransformerModel,
        "통계 기반 모델": StatisticalModel,
        "앙상블 모델": EnsembleModel
    }

def get_model(model_type: str) -> BaseModel:
    """모델 타입에 따른 모델 인스턴스 반환"""
    models = get_available_models()
    if model_type not in models:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")
    return models[model_type]()
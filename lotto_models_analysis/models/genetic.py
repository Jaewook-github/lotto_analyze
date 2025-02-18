
import numpy as np
from .base import BaseModel
import logging
from datetime import datetime


class GeneticAlgorithmModel(BaseModel):
    """유전 알고리즘 기반 로또 번호 예측 모델"""

    def __init__(self, population_size=100, elite_size=4, mutation_rate=0.1):
        super().__init__()
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.population = None
        self.best_fitness_history = []
        self.generation_count = 0
        self.historical_data = None
        self.setup_logging()

    def build_model(self):
        """모델 구조 생성 - 유전 알고리즘은 별도의 신경망 모델이 필요 없음"""
        pass


    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            filename=f'logs/genetic_{datetime.now():%Y%m%d_%H%M%S}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )

    def initialize_population(self):
        """초기 개체군 생성"""
        try:
            population = []
            for _ in range(self.population_size):
                # 1부터 45까지의 숫자 중 6개를 무작위로 선택
                chromosome = np.sort(np.random.choice(45, 6, replace=False) + 1)
                population.append(chromosome)

            logging.info(f"초기 개체군 생성 완료: {self.population_size}개 염색체")
            return population

        except Exception as e:
            logging.error(f"초기 개체군 생성 중 오류: {str(e)}")
            raise

    # def calculate_fitness(self, chromosome):
    #     """적합도 계산"""
    #     try:
    #         fitness = 0
    #         recent_draws = self.historical_data[-50:]  # 최근 50회차만 사용
    #
    #         # 과거 당첨 번호와의 매칭 점수
    #         for numbers in recent_draws:
    #             matches = len(set(chromosome) & set(numbers))
    #             if matches == 6:
    #                 fitness += 1000
    #             elif matches == 5:
    #                 fitness += 100
    #             elif matches == 4:
    #                 fitness += 10
    #             elif matches == 3:
    #                 fitness += 1
    #
    #         # 번호 분포 평가
    #         sections = np.zeros(5)  # 1-9, 10-19, 20-29, 30-39, 40-45
    #         for num in chromosome:
    #             sections[(num - 1) // 10] += 1
    #
    #         # 너무 치우친 분포에 페널티
    #         if max(sections) > 3:
    #             fitness *= 0.8
    #
    #         # 연속된 번호에 대한 페널티
    #         diffs = np.diff(chromosome)
    #         if np.any(diffs == 1):
    #             fitness *= 0.9
    #
    #         return fitness
    #
    #     except Exception as e:
    #         logging.error(f"적합도 계산 중 오류: {str(e)}")
    #         raise

    def calculate_fitness(self, chromosome):
        """적합도 계산"""
        try:
            fitness = 0
            recent_draws = self.historical_data[-30:]  # 최근 30회차만 사용

            # 과거 당첨 번호와의 매칭 점수
            matched_counts = []
            for numbers in recent_draws:
                matches = len(set(chromosome) & set(numbers))
                matched_counts.append(matches)
                if matches == 6:
                    fitness += 100
                elif matches == 5:
                    fitness += 50
                elif matches == 4:
                    fitness += 10
                elif matches == 3:
                    fitness += 1

            # 너무 자주 매칭되는 경우 페널티
            if max(matched_counts) > 4:
                fitness *= 0.5

            # 번호 간격 평가
            gaps = np.diff(sorted(chromosome))
            if np.any(gaps == 1):  # 연속된 번호가 있으면 페널티
                fitness *= 0.8

            # 구간 분포 평가
            sections = np.zeros(5)
            for num in chromosome:
                sections[(num - 1) // 10] += 1

            # 구간 균형 보너스
            if max(sections) <= 2:  # 한 구간에 2개 이하
                fitness *= 1.2

            # 홀짝 균형 평가
            even_count = sum(1 for x in chromosome if x % 2 == 0)
            if 2 <= even_count <= 4:  # 홀짝 균형이 좋으면 보너스
                fitness *= 1.1

            # 합계 범위 평가 (135-185가 적정)
            total = sum(chromosome)
            if 135 <= total <= 185:
                fitness *= 1.1

            return fitness

        except Exception as e:
            logging.error(f"적합도 계산 중 오류: {str(e)}")
            raise

    def select_parents(self, population, fitness_scores):
        """토너먼트 선택으로 부모 선택"""
        try:
            tournament_size = 5
            selected_parents = []

            for _ in range(2):
                tournament_indices = np.random.choice(
                    len(population),
                    tournament_size,
                    replace=False
                )
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_parents.append(population[winner_idx])

            return selected_parents

        except Exception as e:
            logging.error(f"부모 선택 중 오류: {str(e)}")
            raise

    def crossover(self, parent1, parent2):
        """교차 연산"""
        try:
            if np.random.random() < 0.9:  # 교차 확률 90%
                # 교차점 선택
                crossover_point = np.random.randint(1, 5)

                # 자식 생성
                child = np.concatenate([parent1[:crossover_point],
                                        parent2[crossover_point:]])

                # 중복 제거 및 부족한 번호 보충
                unique_numbers = np.unique(child)
                while len(unique_numbers) < 6:
                    new_number = np.random.randint(1, 46)
                    if new_number not in unique_numbers:
                        unique_numbers = np.append(unique_numbers, new_number)

                return np.sort(unique_numbers[:6])
            else:
                return parent1.copy()

        except Exception as e:
            logging.error(f"교차 연산 중 오류: {str(e)}")
            raise

    def mutate(self, chromosome):
        """돌연변이 연산"""
        try:
            if np.random.random() < self.mutation_rate:
                # 무작위 위치 선택
                mutation_point = np.random.randint(0, 6)

                # 새로운 번호 선택 (기존 번호와 중복되지 않게)
                available_numbers = list(set(range(1, 46)) - set(chromosome))
                new_number = np.random.choice(available_numbers)

                # 번호 교체
                chromosome[mutation_point] = new_number

                return np.sort(chromosome)
            return chromosome

        except Exception as e:
            logging.error(f"돌연변이 연산 중 오류: {str(e)}")
            raise

    def evolve(self, generations=100):
        """진화 과정 수행"""
        try:
            if self.population is None:
                self.population = self.initialize_population()

            logging.info(f"진화 시작: {generations}세대")

            for generation in range(generations):
                # 적합도 계산
                fitness_scores = [self.calculate_fitness(chromosome)
                                  for chromosome in self.population]

                # 최고 적합도 기록
                best_fitness = max(fitness_scores)
                self.best_fitness_history.append(best_fitness)

                # 새로운 개체군 생성
                new_population = []

                # 엘리트 보존
                elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
                new_population.extend([self.population[i].copy()
                                       for i in elite_indices])

                # 나머지 개체 생성
                while len(new_population) < self.population_size:
                    # 부모 선택
                    parents = self.select_parents(self.population, fitness_scores)

                    # 자식 생성
                    child = self.crossover(parents[0], parents[1])
                    child = self.mutate(child)

                    new_population.append(child)

                self.population = new_population
                self.generation_count += 1

                # 진행 상황 로깅
                if generation % 10 == 0:
                    logging.info(
                        f"세대 {generation}/{generations}, "
                        f"최고 적합도: {best_fitness:.2f}"
                    )

            logging.info("진화 완료")

        except Exception as e:
            logging.error(f"진화 과정 중 오류: {str(e)}")
            raise

    def train(self, train_data, validation_data=None):
        try:
            self.historical_data = train_data
            self.evolve(generations=100)

            # 최고 성능의 개체군 저장
            best_population = {
                'population': np.array(self.population).tolist() if isinstance(self.population,
                                                                               list) else self.population.tolist(),
                'generation_count': self.generation_count,
                'best_fitness_history': self.best_fitness_history
            }

            np.save('best_models/genetic_model.npy', best_population)


        except Exception as e:
            logging.error(f"학습 중 오류: {str(e)}")
            raise

    # def predict_numbers(self, input_data=None, n_predictions=6):
    #     """번호 예측"""
    #     try:
    #         if not self.population:
    #             raise ValueError("모델이 학습되지 않았습니다.")
    #
    #         # 적합도 기준으로 상위 n_predictions개의 염색체 선택
    #         fitness_scores = [self.calculate_fitness(chromosome)
    #                           for chromosome in self.population]
    #         best_indices = np.argsort(fitness_scores)[-n_predictions:]
    #         predictions = [self.population[i] for i in best_indices]
    #
    #         # 각 예측에 대한 적합도 로깅
    #         for i, pred in enumerate(predictions):
    #             logging.info(
    #                 f"예측 {i + 1}: {pred}, "
    #                 f"적합도: {fitness_scores[best_indices[i]]:.2f}"
    #             )
    #
    #         return predictions[0]  # 가장 높은 적합도를 가진 염색체 반환
    #
    #     except Exception as e:
    #         logging.error(f"예측 중 오류: {str(e)}")
    #         raise

    def predict_numbers(self, input_data=None, n_predictions=6):
        """번호 예측"""
        try:
            predictions = []
            for set_num in range(n_predictions):
                # 매 세트마다 새로운 개체군으로 시작
                self.population = self.initialize_population()

                # 짧은 진화 수행
                self.evolve(generations=20)

                # 적합도 계산
                fitness_scores = [self.calculate_fitness(chromosome)
                                  for chromosome in self.population]

                # 상위 10개의 해 중에서 랜덤 선택
                top_k = 10
                top_indices = np.argsort(fitness_scores)[-top_k:]
                selected_idx = np.random.choice(top_indices)
                selected_numbers = list(map(int, self.population[selected_idx]))

                # 돌연변이 확률 적용
                if np.random.random() < 0.3:  # 30% 확률로 돌연변이 발생
                    mutation_idx = np.random.randint(6)
                    available = list(set(range(1, 46)) - set(selected_numbers))
                    selected_numbers[mutation_idx] = np.random.choice(available)
                    selected_numbers.sort()

                predictions.append(selected_numbers)
                logging.info(f"세트 {set_num + 1} 예측 완료: {selected_numbers}")

            return predictions[0]  # 첫 번째 예측 반환

        except Exception as e:
            logging.error(f"예측 중 오류: {str(e)}")
            raise

    def get_population_stats(self):
        """개체군 통계 정보 반환"""
        try:
            if not self.population:
                return None

            fitness_scores = [self.calculate_fitness(chromosome)
                              for chromosome in self.population]

            return {
                'generation': self.generation_count,
                'population_size': len(self.population),
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'worst_fitness': min(fitness_scores),
                'fitness_std': np.std(fitness_scores)
            }

        except Exception as e:
            logging.error(f"통계 계산 중 오류: {str(e)}")
            raise

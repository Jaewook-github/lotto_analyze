
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from .base import BaseModel
import logging


class ReinforcementLearningModel(BaseModel):
    """강화학습 기반 로또 번호 예측 모델"""

    def __init__(self):
        super().__init__()
        # 강화학습 파라미터
        self.state_size = 45  # 상태 공간 크기 (로또 번호 범위)
        self.action_size = 45  # 행동 공간 크기 (선택 가능한 번호)
        self.memory = deque(maxlen=2000)  # 경험 메모리
        self.gamma = 0.95  # 할인 계수
        self.epsilon = 1.0  # 탐험률
        self.epsilon_min = 0.01  # 최소 탐험률
        self.epsilon_decay = 0.995  # 탐험률 감소율
        self.learning_rate = 0.001  # 학습률

        # 모델 초기화
        self.model = self.build_model()  # 메인 네트워크
        self.target_model = self.build_model()  # 타겟 네트워크
        self.update_target_model()  # 타겟 네트워크 초기화

    def softmax(self, x):
        """소프트맥스 함수"""
        e_x = np.exp(x - np.max(x))  # 오버플로우 방지
        return e_x / e_x.sum()


    def build_model(self):
        """DQN 모델 구축"""
        try:
            model = Sequential([
                Dense(256, input_dim=self.state_size, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dense(self.action_size, activation='linear')
            ])

            model.compile(
                loss='mse',
                optimizer=Adam(learning_rate=self.learning_rate)
            )

            self.model = model  # 모델을 인스턴스 변수로 저장
            return model

        except Exception as e:
            logging.error(f"모델 구축 중 오류 발생: {str(e)}")
            raise

    def update_target_model(self):
        """타겟 네트워크 업데이트"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=False):
        """행동 선택 (번호 선택)"""
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """경험 재생을 통한 학습"""
        try:
            if len(self.memory) < batch_size:
                return

            minibatch = random.sample(self.memory, batch_size)
            states = np.zeros((batch_size, self.state_size))
            next_states = np.zeros((batch_size, self.state_size))

            # 배치 데이터 준비
            for i, (state, _, _, next_state, _) in enumerate(minibatch):
                states[i] = state
                next_states[i] = next_state

            # 현재 상태와 다음 상태의 Q값 예측
            target = self.model.predict(states, verbose=0)
            target_next = self.target_model.predict(next_states, verbose=0)

            # Q-learning 업데이트
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                if done:
                    target[i][action] = reward
                else:
                    target[i][action] = reward + self.gamma * np.amax(target_next[i])

            # 모델 학습
            self.model.fit(states, target, epochs=1, verbose=0)

            # 탐험률 감소
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        except Exception as e:
            logging.error(f"경험 재생 중 오류 발생: {str(e)}")
            raise

    def calculate_reward(self, selected_numbers, winning_numbers):
        """보상 계산"""
        matches = len(set(selected_numbers) & set(winning_numbers))

        # 매칭 개수에 따른 보상
        rewards = {
            6: 1000,  # 1등
            5: 100,  # 3등 (보너스 번호 제외)
            4: 10,  # 4등
            3: 1,  # 5등
            2: 0,  # 미당첨
            1: -1,  # 페널티
            0: -2  # 페널티
        }

        return rewards.get(matches, -2)

    # def train_episode(self, historical_data, episode_count=1000):
    #     """에피소드 단위 학습"""
    #     try:
    #         logging.info(f"강화학습 시작 - {episode_count}개 에피소드")
    #
    #         for episode in range(episode_count):
    #             state = np.zeros(self.state_size)
    #             total_reward = 0
    #             selected_numbers = []
    #
    #             # 6개 번호 선택
    #             for step in range(6):
    #                 # 행동 선택 (번호 선택)
    #                 action = self.act(state.reshape(1, -1), training=True)
    #
    #                 # 이미 선택된 번호는 제외
    #                 while action + 1 in selected_numbers:
    #                     action = np.random.randint(self.action_size)
    #
    #                 selected_numbers.append(action + 1)
    #                 next_state = state.copy()
    #                 next_state[action] = 1
    #
    #                 # 보상 계산
    #                 if len(selected_numbers) == 6:
    #                     # 최근 당첨 번호들과 비교
    #                     rewards = []
    #                     for winning_numbers in historical_data[-10:]:
    #                         reward = self.calculate_reward(
    #                             selected_numbers, winning_numbers
    #                         )
    #                         rewards.append(reward)
    #                     reward = np.mean(rewards)  # 평균 보상 사용
    #                 else:
    #                     reward = 0
    #
    #                 done = (step == 5)
    #                 self.remember(state, action, reward, next_state, done)
    #                 state = next_state
    #                 total_reward += reward
    #
    #                 # 경험 재생
    #                 if len(self.memory) > 32:
    #                     self.replay(32)
    #
    #             # 주기적으로 타겟 네트워크 업데이트
    #             if episode % 10 == 0:
    #                 self.update_target_model()
    #
    #             # 학습 진행 상황 로깅
    #             if episode % 100 == 0:
    #                 logging.info(
    #                     f"Episode: {episode}/{episode_count}, "
    #                     f"Reward: {total_reward:.2f}, "
    #                     f"Epsilon: {self.epsilon:.4f}"
    #                 )
    #
    #         logging.info("강화학습 완료")
    #
    #     except Exception as e:
    #         logging.error(f"학습 중 오류 발생: {str(e)}")
    #         raise
    #
    # def train(self, train_data, validation_data=None):
    #     """학습 인터페이스 구현"""
    #     return self.train_episode(train_data)

    def train_episode(self, historical_data, episode_count=1000):
        """에피소드 단위 학습"""
        try:
            # 데이터 설정
            self.historical_data = historical_data
            batch_size = 64
            total_reward_history = []

            logging.info("강화학습 시작...")
            logging.info(f"학습 데이터 크기: {len(historical_data)} 회차")
            logging.info(f"배치 크기: {batch_size}")
            logging.info(f"에피소드 수: {episode_count}")

            # 에피소드 단위 학습
            for episode in range(episode_count):
                total_reward = 0
                state = np.zeros(self.state_size)
                selected_numbers = []

                # 번호 6개 선택
                for _ in range(6):
                    action = self.act(state.reshape(1, -1), training=True)
                    while action + 1 in selected_numbers:
                        action = np.random.randint(self.action_size)

                    selected_numbers.append(action + 1)
                    next_state = state.copy()
                    next_state[action] = 1

                    # 보상 계산
                    if len(selected_numbers) == 6:
                        rewards = []
                        for winning_numbers in historical_data[-10:]:
                            reward = self.calculate_reward(selected_numbers, winning_numbers)
                            rewards.append(reward)
                        reward = np.mean(rewards)
                    else:
                        reward = 0

                    done = (len(selected_numbers) == 6)
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward

                # 경험 재생으로 학습
                if len(self.memory) > batch_size:
                    self.replay(batch_size)

                # 타겟 네트워크 업데이트
                if episode % 10 == 0:
                    self.update_target_model()

                # 탐험률 감소
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                total_reward_history.append(total_reward)

                # 학습 진행 상황 로깅
                if episode % 100 == 0:
                    avg_reward = np.mean(total_reward_history[-100:])
                    logging.info(
                        f"Episode: {episode}/{episode_count}, "
                        f"Average Reward: {avg_reward:.2f}, "
                        f"Epsilon: {self.epsilon:.4f}"
                    )

            # 최고 성능의 모델 저장
            self.model.save('best_models/reinforcement_model.keras')
            logging.info("모델 저장 완료: best_models/reinforcement_model.keras")

            return total_reward_history

        except Exception as e:
            logging.error(f"학습 중 오류 발생: {str(e)}")
            raise

    def train(self, train_data, validation_data=None):
        """학습 인터페이스 구현"""
        try:
            # train_episode 호출하여 학습 수행
            reward_history = self.train_episode(train_data, episode_count=1000)

            # 학습 결과 로깅
            final_avg_reward = np.mean(reward_history[-100:])
            logging.info("\n=== 학습 완료 ===")
            logging.info(f"최종 평균 보상: {final_avg_reward:.2f}")
            logging.info(f"최종 탐험률: {self.epsilon:.4f}")
            logging.info(f"메모리 크기: {len(self.memory)}")

            return reward_history

        except Exception as e:
            logging.error(f"학습 중 오류 발생: {str(e)}")
            raise

    # 강화 학습 모델_1
    # def predict_numbers(self, input_data=None, n_predictions=6):
    #     """번호 예측"""
    #     try:
    #         if not self.model:
    #             raise ValueError("모델이 학습되지 않았습니다.")
    #
    #         # 각 예측마다 새로운 상태로 시작
    #         eps = 0.3  # 약간의 무작위성 추가
    #         selected_numbers = []
    #         max_attempts = 100
    #
    #         # 6개의 번호 선택
    #         while len(selected_numbers) < n_predictions and max_attempts > 0:
    #             # epsilon-greedy 전략 사용
    #             if np.random.random() < eps:
    #                 # 무작위 선택
    #                 available_numbers = list(set(range(1, 46)) - set(selected_numbers))
    #                 number = int(np.random.choice(available_numbers))
    #                 selected_numbers.append(number)
    #             else:
    #                 # Q값 기반 선택
    #                 state = np.zeros(self.state_size)
    #                 for num in selected_numbers:
    #                     state[num - 1] = 1
    #
    #                 state_reshaped = state.reshape(1, -1)
    #                 q_values = self.model.predict(state_reshaped, verbose=0)[0]
    #
    #                 # 이미 선택된 번호 마스킹
    #                 available_actions = list(set(range(self.action_size)) -
    #                                          set(x - 1 for x in selected_numbers))
    #
    #                 if not available_actions:
    #                     break
    #
    #                 # 상위 5개 액션 중에서 랜덤 선택
    #                 masked_q_values = q_values[available_actions]
    #                 top_k = min(5, len(available_actions))
    #                 top_indices = np.argsort(masked_q_values)[-top_k:]
    #                 selected_idx = np.random.choice(top_indices)
    #                 action = available_actions[selected_idx]
    #                 number = int(action + 1)
    #                 selected_numbers.append(number)
    #
    #             max_attempts -= 1
    #
    #         # 부족한 번호 채우기
    #         if len(selected_numbers) < n_predictions:
    #             available_numbers = list(set(range(1, 46)) - set(selected_numbers))
    #             remaining_count = n_predictions - len(selected_numbers)
    #             additional_numbers = list(np.random.choice(
    #                 available_numbers,
    #                 size=remaining_count,
    #                 replace=False
    #             ))
    #             selected_numbers.extend(additional_numbers)
    #
    #         # 결과 검증 및 정렬
    #         result = sorted(list(map(int, selected_numbers)))
    #         logging.info(f"예측된 번호: {result}")
    #
    #         return result
    #
    #     except Exception as e:
    #         logging.error(f"예측 중 오류: {str(e)}")
    #         raise

    def predict_numbers(self, input_data=None, n_sets=1):
        """번호 예측"""
        try:
            if not self.model:
                raise ValueError("모델이 학습되지 않았습니다.")

            # 전체 번호에 대한 Q값 예측
            state = np.zeros((1, self.state_size))
            q_values = self.model.predict(state, verbose=0)[0]

            # Q값을 기반으로 상위 20개의 유망한 번호 선택
            top_k = 20
            top_indices = np.argsort(q_values)[-top_k:]
            top_numbers = [int(i + 1) for i in top_indices]

            logging.info(f"학습된 상위 {top_k}개 번호: {top_numbers}")

            # 각 세트별로 다른 번호 조합 생성
            predictions = []
            for set_num in range(n_sets):  # n_sets 만큼 생성
                # 상위 번호들 중에서 6개 선택
                available = top_numbers.copy()
                selected = []

                # Q값에 기반한 가중치 계산
                if hasattr(self, 'softmax'):
                    weights = self.softmax(q_values[np.array(top_indices)])
                else:
                    # 정적 메서드인 경우
                    weights = self.__class__.softmax(q_values[np.array(top_indices)])

                # 6개 번호 선택
                while len(selected) < 6 and available:
                    available_indices = [i for i, num in enumerate(top_numbers) if num in available]
                    if not available_indices:
                        break

                    available_weights = weights[available_indices]
                    available_weights = available_weights / np.sum(available_weights)

                    selected_idx = np.random.choice(available_indices, p=available_weights)
                    selected_number = top_numbers[selected_idx]

                    selected.append(selected_number)
                    available.remove(selected_number)

                # 결과 정렬
                selected = sorted(selected)
                predictions.append(selected)
                logging.info(f"세트 {set_num + 1} 예측 완료: {selected}")

            # n_sets개의 예측 결과를 모두 반환
            if len(predictions) == 1:
                return predictions[0]
            return predictions[0]  # 호환성을 위해 첫 번째 세트만 반환

        except Exception as e:
            logging.error(f"예측 중 오류: {str(e)}")
            raise


    def get_model_status(self):
        """모델 상태 정보 반환"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'learning_rate': self.learning_rate
        }

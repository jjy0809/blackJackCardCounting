import random  # 무작위 동작을 위해 random 모듈 임포트
import torch  # PyTorch 임포트 (딥러닝 프레임워크)
import torch.nn as nn  # PyTorch의 신경망 모듈 임포트
import torch.optim as optim  # PyTorch의 최적화 알고리즘 모듈 임포트
from collections import deque  # 경험 리플레이를 위한 deque 자료구조 임포트
from tqdm import tqdm  # 학습 진행 상황을 시각적으로 표시하기 위해 tqdm 임포트
import numpy as np  # 수치 계산을 위해 numpy 임포트

# 블랙잭 게임 설정
seed = 1000  # 초기 자본금 설정
C_PER_SET = 52  # 카드 세트당 카드 수
D_STAND = 17  # 딜러 스탠드 기준 점수
CARD = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']  # 카드 종류 정의
P_CARD = ['A', 'J', 'Q', 'K']  # 10점 카드 정의
learning_rate = 0.0005  # 학습률 설정
gamma = 0.998  # 할인율 설정
eps_start = 1.0  # 탐험 비율 초기값
eps_end = 0.01  # 탐험 비율 최소값
eps_decay = 0.999  # 탐험 비율 감소율
batch_size = 128  # 학습 배치 크기
target_update = 200  # 타겟 네트워크 업데이트 주기
num_episodes = 1000000  # 총 에피소드 수

BET_OPTIONS = [10, 20, 30, 50, 70, 100, 150, 200, 250, 400, 500]  # 베팅 옵션

# Q-네트워크 정의
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):  # 입력 및 출력 크기 설정
        super(QNetwork, self).__init__()  # 부모 클래스 초기화
        self.fc1 = nn.Linear(state_size, 128)  # 첫 번째 완전 연결 계층
        self.fc2 = nn.Linear(128, 64)  # 두 번째 완전 연결 계층
        self.fc3 = nn.Linear(64, action_size)  # 출력 계층

    def forward(self, x):  # 순전파 정의
        x = torch.relu(self.fc1(x))  # 첫 번째 계층에 ReLU 활성화 함수 적용
        x = torch.relu(self.fc2(x))  # 두 번째 계층에 ReLU 활성화 함수 적용
        return self.fc3(x)  # 출력 반환

# 강화학습 환경 (블랙잭 게임 규칙 포함)
class BlackjackEnv:
    def __init__(self):  # 환경 초기화
        self.money = seed  # 초기 자본금 설정
        self.c_set = 3  # 카드 세트 수 설정
        self.cards = []  # 카드 덱 초기화
        self.card_count = 0  # 카드카운팅 변수 초기화
        self.shuffle()  # 카드 덱 셔플
        self.reset()  # 환경 초기화

    def shuffle(self):  # 카드 덱을 셔플하는 함수
        self.cards = self.c_set * CARD * 4  # 카드 덱 생성
        random.shuffle(self.cards)  # 덱 셔플
        self.card_count = 0  # 카드카운트 초기화

    def pick_card(self):  # 카드 덱에서 한 장을 뽑는 함수
        if not self.cards:  # 덱이 비어있는 경우
            return None  # None 반환
        card = self.cards.pop()  # 카드 한 장 뽑기
        # 카드카운팅 업데이트
        if card in [2, 3, 4, 5, 6]:
            self.card_count += 1  # 낮은 카드로 카운트 증가
        elif card in [10, 'J', 'Q', 'K', 'A']:
            self.card_count -= 1  # 높은 카드로 카운트 감소
        return card  # 뽑은 카드 반환

    def reset(self):  # 에피소드 초기화 함수
        self.money = seed  # 초기 자본금 설정
        self.player_hand = []  # 플레이어 핸드 초기화
        self.dealer_hand = []  # 딜러 핸드 초기화
        self.shuffle()  # 덱 셔플
        return self.get_state()  # 초기 상태 반환

    def get_state(self):  # 현재 상태 반환 함수
        player_sum = self.calculate_hand(self.player_hand)  # 플레이어 카드 합산
        dealer_card = self.dealer_hand[0] if self.dealer_hand else 0  # 딜러의 오픈 카드
        return torch.tensor([self.money, player_sum, dealer_card, self.card_count], dtype=torch.float32)  # 상태 텐서 반환

    def calculate_hand(self, hand):  # 카드 합산 함수
        total = 0  # 총합 초기화
        ace_count = 0  # 에이스 개수 초기화
        for card in hand:  # 각 카드에 대해 합산
            if card in P_CARD:
                total += 10  # 10점 카드
            elif card == 'A':
                total += 11  # 에이스는 11점으로 합산
                ace_count += 1  # 에이스 개수 증가
            else:
                total += card  # 숫자 카드 합산
        while total > 21 and ace_count:  # 21을 초과하고 에이스가 있는 경우
            total -= 10  # 에이스를 1로 변경
            ace_count -= 1  # 에이스 개수 감소
        return total  # 총합 반환

    def step(self, action):  # 게임의 한 스텝 진행
        bet_option = action // 2  # 베팅 옵션 선택
        if bet_option >= len(BET_OPTIONS):  # 베팅 옵션 범위 초과 시 예외 발생
            raise IndexError("인덱스 애러")  # 에러 발생

        requested_bet = BET_OPTIONS[bet_option]  # 선택된 베팅 금액
        bet = min(requested_bet, self.money)  # 자본금보다 큰 베팅 방지
        move = action % 2  # 0: 스탠드, 1: 힛
        initial_money = self.money  # 초기 자본금 저장

        if move == 1:  # 힛 선택 시
            card = self.pick_card()  # 카드 한 장 뽑기
            if card is None:  # 덱 소진 시
                self.money = initial_money  # 자본금 초기화
                self.reset_game()  # 게임 리셋
                return self.get_state(), "invalid"  # 상태 반환
            self.player_hand.append(card)  # 플레이어 핸드에 추가
            if self.calculate_hand(self.player_hand) > 21:  # 버스트 시
                self.money = max(0, self.money - bet)  # 자본금에서 베팅 금액 차감
                self.reset_game()  # 게임 리셋
                return self.get_state(), "bust"  # 상태 반환
        else:  # 스탠드 선택 시
            while self.calculate_hand(self.dealer_hand) < D_STAND:  # 딜러의 합이 17 미만일 경우
                card = self.pick_card()  # 카드 한 장 뽑기
                if card is None:  # 덱 소진 시
                    self.money = initial_money  # 자본금 초기화
                    self.reset_game()  # 게임 리셋
                    return self.get_state(), "invalid"  # 상태 반환
                self.dealer_hand.append(card)  # 딜러 핸드에 추가
            dealer_sum = self.calculate_hand(self.dealer_hand)  # 딜러 카드 합산
            player_sum = self.calculate_hand(self.player_hand)  # 플레이어 카드 합산
            if player_sum > dealer_sum or dealer_sum > 21:  # 승리 조건
                self.money += bet  # 자본금에 베팅 금액 추가
                result = "win"  # 승리 설정
            elif player_sum < dealer_sum:  # 패배 조건
                self.money = max(0, self.money - bet)  # 자본금에서 베팅 금액 차감
                result = "loss"  # 패배 설정
            else:  # 무승부
                result = "draw"  # 무승부 설정
            self.reset_game()  # 게임 리셋
            return self.get_state(), result  # 상태 반환

        return self.get_state(), "continue"  # 계속 상태 반환

    def reset_game(self):  # 게임 초기화 함수
        self.player_hand = []  # 플레이어 핸드 초기화
        self.dealer_hand = []  # 딜러 핸드 초기화

    def is_done(self):  # 에피소드 종료 조건
        return self.money < 10 or self.money >= 2000 or not self.cards  # 조건 반환

# DQN 에이전트 정의
class DQNAgent:
    def __init__(self):  # 에이전트 초기화
        self.policy_net = QNetwork(4, len(BET_OPTIONS) * 2)  # 정책 네트워크 생성
        self.target_net = QNetwork(4, len(BET_OPTIONS) * 2)  # 타겟 네트워크 생성
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 타겟 네트워크 동기화
        self.target_net.eval()  # 타겟 네트워크 평가 모드 설정
        self.memory = deque(maxlen=10000)  # 경험 리플레이 메모리 설정
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # Adam 최적화 알고리즘 설정
        self.steps_done = 0  # 탐험 단계 초기화
        self.epsilon = eps_start  # 탐험 비율 초기화

    def select_action(self, state):  # 행동 선택 함수
        sample = random.random()  # 무작위 수 생성
        action_space_size = len(BET_OPTIONS) * 2  # 행동 공간 크기
        if sample > self.epsilon:  # 탐험 비율보다 큰 경우
            with torch.no_grad():  # 탐색 없이 행동 선택
                return self.policy_net(state).argmax().item()  # 최적의 행동 반환
        else:  # 탐험 비율보다 작은 경우
            return random.randrange(action_space_size)  # 무작위 행동 반환

    def optimize(self):  # 네트워크 최적화 함수
        if len(self.memory) < batch_size:  # 메모리 크기 확인
            return  # 충분하지 않으면 학습 종료
        transitions = random.sample(self.memory, batch_size)  # 무작위 샘플링
        batch = list(zip(*transitions))  # 배치 재구성
        state_batch = torch.stack(batch[0])  # 상태 배치
        action_batch = torch.tensor(batch[1])  # 행동 배치
        reward_batch = torch.tensor(batch[2])  # 보상 배치
        next_state_batch = torch.stack(batch[3])  # 다음 상태 배치
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), dtype=torch.bool)  # 유효 상태 마스크

        # 현재 상태-행동 값 계산
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze()
        next_state_values = torch.zeros(batch_size)  # 다음 상태 값 초기화

        # 다음 상태 값 계산 (비어 있지 않은 상태에 대해 최대 값 사용)
        next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].detach()

        # 기대값 계산(몬테카를로 방법 적용)
        expected_state_action_values = torch.mean((next_state_values * gamma) + reward_batch)

        # 손실 함수 계산 (MSE 사용)
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.repeat(batch_size))

        # 경사 초기화
        self.optimizer.zero_grad()
        loss.backward()  # 역전파 수행
        self.optimizer.step()  # 가중치 업데이트

        # 탐험 비율 업데이트
        self.epsilon = max(eps_end, self.epsilon * eps_decay)

def train():  # 학습 함수
    env = BlackjackEnv()  # 환경 초기화
    agent = DQNAgent()  # 에이전트 초기화

    for e in tqdm(range(num_episodes), desc="Training Progress"):  # 학습 진행 시각화
        state = env.reset()  # 에피소드 초기 상태
        initial_money = env.money  # 초기 자본금 기록
        episode_experiences = []  # 에피소드 내 경험 저장

        while not env.is_done():  # 에피소드 종료 조건까지 반복
            action = agent.select_action(state)  # 에이전트가 행동 선택
            next_state, result = env.step(action)  # 선택한 행동으로 환경 진행

            if env.money != initial_money:  # 자본금 변화가 있는 경우
                reward = (env.money - initial_money) / initial_money * 100  # 보상 계산
                episode_experiences.append((state, action, reward, next_state))  # 경험 저장

            state = next_state  # 상태 업데이트

        # 에피소드의 모든 경험을 메모리에 추가
        for state, action, reward, next_state in episode_experiences:
            agent.memory.append((state, action, reward, next_state))
        agent.optimize()  # 네트워크 최적화 수행

        # 일정 주기마다 타겟 네트워크 업데이트
        if e % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    # 모델 저장 경로
    model_path = r"C:\Users\happy\Desktop\model\integ\model_1.pt"
    torch.save(agent.policy_net.state_dict(), model_path)  # 모델 저장
    print(f"{model_path}")  # 모델 저장 확인 메시지

if __name__ == '__main__':
    train()  # 학습 함수 호출

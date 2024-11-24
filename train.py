import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

# 블랙잭 게임 설정
seed = 1000
C_PER_SET = 52
D_STAND = 17
CARD = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
P_CARD = ['A', 'J', 'Q', 'K']
learning_rate = 0.0005
gamma = 0.998
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.999
batch_size = 128
target_update = 200
num_episodes = 1000000

BET_OPTIONS = [10, 20, 30, 50, 70, 100, 150, 200, 250, 400, 500]

# Q-네트워크 정의
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 강화학습 환경 
class BlackjackEnv:
    def __init__(self):
        self.money = seed
        self.c_set = 3
        self.cards = []
        self.card_count = 0
        self.shuffle()
        self.reset()

    def shuffle(self):
        self.cards = self.c_set * CARD * 4
        random.shuffle(self.cards)
        self.card_count = 0

    def pick_card(self):
        if not self.cards:
            return None
        card = self.cards.pop()
        if card in [2, 3, 4, 5, 6]:
            self.card_count += 1
        elif card in [10, 'J', 'Q', 'K', 'A']:
            self.card_count -= 1
        return card

    def reset(self):
        self.money = seed
        self.player_hand = []
        self.dealer_hand = []
        self.shuffle()
        return self.get_state()

    def get_state(self):
        player_sum = self.calculate_hand(self.player_hand)
        dealer_card = self.dealer_hand[0] if self.dealer_hand else 0
        return torch.tensor([self.money, player_sum, dealer_card, self.card_count], dtype=torch.float32)

    def calculate_hand(self, hand):
        total = 0
        ace_count = 0
        for card in hand:
            if card in P_CARD:
                total += 10
            elif card == 'A':
                total += 11
                ace_count += 1
            else:
                total += card
        while total > 21 and ace_count:
            total -= 10
            ace_count -= 1
        return total

    def step(self, action):
        bet_option = action // 2
        if bet_option >= len(BET_OPTIONS):
            raise IndexError("인덱스 애러")

        requested_bet = BET_OPTIONS[bet_option]
        bet = min(requested_bet, self.money)
        move = action % 2
        initial_money = self.money

        if move == 1:  # 힛
            card = self.pick_card()
            if card is None:
                self.money = initial_money
                self.reset_game()
                return self.get_state(), "invalid"
            self.player_hand.append(card)
            if self.calculate_hand(self.player_hand) > 21:
                self.money = max(0, self.money - bet)
                self.reset_game()
                return self.get_state(), "bust"
        else:  # 스탠드
            while self.calculate_hand(self.dealer_hand) < D_STAND:
                card = self.pick_card()
                if card is None:
                    self.money = initial_money
                    self.reset_game()
                    return self.get_state(), "invalid"
                self.dealer_hand.append(card)
            dealer_sum = self.calculate_hand(self.dealer_hand)
            player_sum = self.calculate_hand(self.player_hand)
            if player_sum > dealer_sum or dealer_sum > 21:
                self.money += bet
                result = "win"
            elif player_sum < dealer_sum:
                self.money = max(0, self.money - bet)
                result = "loss"
            else:
                result = "draw"
            self.reset_game()
            return self.get_state(), result

        return self.get_state(), "continue"

    def reset_game(self):
        self.player_hand = []
        self.dealer_hand = []

    def is_done(self):
        return self.money < 10 or self.money >= 2000 or not self.cards

# DQN 에이전트 정의
class DQNAgent:
    def __init__(self):
        self.policy_net = QNetwork(4, len(BET_OPTIONS) * 2)
        self.target_net = QNetwork(4, len(BET_OPTIONS) * 2)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.steps_done = 0
        self.epsilon = eps_start

    def select_action(self, state):
        sample = random.random()
        action_space_size = len(BET_OPTIONS) * 2
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(action_space_size)

    def optimize(self):
        if len(self.memory) < batch_size:
            return
        transitions = random.sample(self.memory, batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.stack(batch[0])
        action_batch = torch.tensor(batch[1])
        reward_batch = torch.tensor(batch[2])
        next_state_batch = torch.stack(batch[3])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), dtype=torch.bool)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze()
        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].detach()

        expected_state_action_values = (next_state_values * gamma) + reward_batch
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(eps_end, self.epsilon * eps_decay)

def train():
    env = BlackjackEnv()
    agent = DQNAgent()

    for e in tqdm(range(num_episodes), desc="Training Progress"):
        state = env.reset()
        initial_money = env.money
        episode_experiences = []

        while not env.is_done():
            action = agent.select_action(state)
            next_state, result = env.step(action)

            if env.money != initial_money:
                reward = (env.money - initial_money) / initial_money * 100
                episode_experiences.append((state, action, reward, next_state))

            state = next_state

        for state, action, reward, next_state in episode_experiences:
            agent.memory.append((state, action, reward, next_state))
        agent.optimize()

        if e % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    model_path = r"C:\Users\happy\Desktop\model\count\model_5.pt"
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"{model_path}")

if __name__ == '__main__':
    train()
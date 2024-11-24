import random
import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm

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

# 블랙잭 환경 클래스
class BlackjackEnv:
    def __init__(self):
        self.money = 1000
        self.c_set = 3
        self.cards = []
        self.card_count = 0
        self.shuffle()
        self.reset()
        self.BET_OPTIONS = [10, 20, 30, 50, 70, 100, 150, 200, 250, 400, 500]
        self.stand_sums = []

    def shuffle(self):
        CARD = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
        self.cards = self.c_set * CARD * 4
        random.shuffle(self.cards)
        self.card_count = 0

    def pick_card(self):
        if not self.cards:
            self.shuffle()
        card = self.cards.pop()
        if card in [2, 3, 4, 5, 6]:
            self.card_count += 1
        elif card in [10, 'J', 'Q', 'K', 'A']:
            self.card_count -= 1
        return card

    def reset(self):
        self.money = 1000
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
            if card in ['J', 'Q', 'K']:
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
        if bet_option >= len(self.BET_OPTIONS):
            raise IndexError("Invalid action: bet_option out of range")

        requested_bet = self.BET_OPTIONS[bet_option]
        bet = min(requested_bet, self.money)
        move = action % 2
        initial_money = self.money

        if move == 1:  # 힛
            card = self.pick_card()
            self.player_hand.append(card)
            if self.calculate_hand(self.player_hand) > 21:
                self.money = max(0, self.money - bet)
                self.reset_game()
                return self.get_state(), "bust"
        else:  # 스탠드
            player_sum_on_stand = self.calculate_hand(self.player_hand)
            self.stand_sums.append(player_sum_on_stand)

            while self.calculate_hand(self.dealer_hand) < 17:
                card = self.pick_card()
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
        return self.money < 10

# DQN 에이전트 정의
class DQNAgent:
    def __init__(self, model_path):
        self.policy_net = QNetwork(4, len([10, 20, 30, 50, 70, 100, 150, 200, 250, 400, 500]) * 2)
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()

    def select_action(self, state):
        with torch.no_grad():
            q_values = self.policy_net(state)  # Q-값 계산
            action_probabilities = torch.softmax(q_values, dim=0)  # 행동 확률 분포 계산
            return torch.multinomial(action_probabilities, 1).item()  # 샘플링하여 행동 선택

# 평가 함수
def evaluate_model():
    model_path = r"C:\Users\happy\Desktop\model\integ\model_1.pt"
    log_path = r"C:\Users\happy\Desktop\학교\고등학교\1학년\블랙잭 카드 카운팅\integ\res.txt"

    max_games = int(input("최대 게임 수:  "))

    agent = DQNAgent(model_path)
    env = BlackjackEnv()
    state = env.reset()

    BET_OPTIONS = env.BET_OPTIONS
    bet_counts = Counter({bet: 0 for bet in BET_OPTIONS})

    total_profit = 0
    total_bets = 0
    total_games = 0
    total_wins = 0
    total_draws = 0
    total_busts = 0
    profit_rates = []
    initial_money = 1000

    highest_money = initial_money
    highest_money_game = 0

    for game_number in tqdm(range(1, max_games + 1), desc="Progress", unit="game"):
        if env.is_done():
            break
        total_games += 1
        initial_game_money = env.money

        action = agent.select_action(state)
        next_state, result = env.step(action)

        bet_amount = min(BET_OPTIONS[action // 2], initial_game_money)
        bet_counts[bet_amount] += 1

        if result == "bust":
            total_busts += 1
        elif result == "draw":
            total_draws += 1
        elif result == "win":
            total_wins += 1

        game_profit = env.money - initial_game_money
        profit_rate = (game_profit / initial_game_money) * 100 if initial_game_money > 0 else 0
        profit_rates.append(profit_rate)
        total_profit += game_profit
        total_bets += bet_amount
        state = next_state

        if env.money > highest_money:
            highest_money = env.money
            highest_money_game = game_number

    average_stand_sum = sum(env.stand_sums) / len(env.stand_sums) if env.stand_sums else 0

    average_profit_rate = sum(profit_rates) / len(profit_rates) if profit_rates else 0
    final_profit_rate = ((env.money - initial_money) / initial_money) * 100
    average_bet = total_bets / total_games if total_games > 0 else 0
    win_rate = (total_wins / total_games) * 100 if total_games > 0 else 0

    log = (
        f"\n"
        f"진행한 판 수: {total_games}\n"
        f"최종 남은 돈: {env.money}\n"
        f"최종 수익: {env.money - initial_money}\n"
        f"평균 수익률: {average_profit_rate:.2f}%\n"
        f"최종 수익률: {final_profit_rate:.2f}%\n"
        f"승률: {win_rate:.2f}%\n"
        f"평균 베팅 금액: {average_bet:.2f}\n"
        f"플레이어 버스트 횟수: {total_busts}\n"
        f"무승부 횟수: {total_draws}\n"
        f"각 베팅 금액별 베팅 횟수: {dict(bet_counts)}\n"
        f"최고 자본금: {highest_money} (게임 #{highest_money_game})\n"
        f"스탠드 시 플레이어 카드 합의 평균: {average_stand_sum:.2f}\n"  
        f"\n"
    )

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(f"\n{log}\n")

    print(log)

if __name__ == "__main__":
    evaluate_model()

import random
import math
import sys


f = open('C:/Users/happy/Desktop/새 폴더 (2)/bAI.txt', 'a')


seed = 5000
money = seed
c_set = 1

C_PER_SET = 52
D_STAND = 17  # ~16 HIT
c_arr = []
c_s = random.randint(math.floor(C_PER_SET*c_set/7), math.ceil(C_PER_SET*c_set/5))

'''
A = 11 #or 1
J = 10
Q = 10
K = 10
'''
CARD = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
P_CARD = ['A',  'J', 'Q', 'K']

m_game = 15
game = 1

game_code = 0
'''
-1 Dealer Bust
1 Player bust
2 surrender
3 blackjack
4 push
'''

bet = 0

hidden = 1

win = 0
lose = 0
blj = 0
ave_bet = 0

cm = 0

count = 0

def dp(p):
    #print(p)
    a2 = 0


def shuffle():
    global left_c
    global c_arr
    
    left_c = []
    c_arr = []

    for i in range(c_set):
        for j in range(4):
            for c in CARD:
                left_c.append(c)

    while(len(left_c) != 0):
        r_card = random.choice(left_c)
        left_c.remove(r_card)
        c_arr.append(r_card)

    dp(c_arr)

def pick_c():
    if len(c_arr) <= c_s:
        shuffle()
        print('----- Deck Shuffled -----')
        count = 0
    
    pick = c_arr[0]
    c_arr.remove(pick)
    counting(pick)
    return(c_arr[0])

def sum_c(d, p):
    dsum = 0
    da = 0
    for dc in d:
        if dc in P_CARD:
            if dc == P_CARD[0]:
                dsum += 11
                da += 1
            else:
                dsum += 10
        else:
            dsum += dc
    if da > 0:
        while(dsum > 21 and da > 0):
            dsum -= 10
            da -= 1

    psum = 0
    pa = 0
    for pc in p:
        if pc in P_CARD:
            if pc == P_CARD[0]:
                psum += 11
                pa += 1
            else:
                psum += 10
        else:
            psum += pc
    if pa > 0:
        while(psum > 21 and pa > 0):
            psum -= 10
            pa -= 1
    
    return([dsum, psum])


def print_c(d, p):
    sc = sum_c(d, p)

    print("Dealer's Card: ", end='')
    for i, c in enumerate(d):
        if i == 1 and hidden == 1:
            print('?', end='')
        else:
            print(c, end='')
        print(' ', end='')
    if hidden == 1:
        print(' = ' + '?')
    else:
        print(' = ' + str(sc[0]))

    print("Player's Card: ", end='')
    for c in p:
        print(c, end='')
        print(' ', end='')
    print(' = ' + str(sc[1]))
    printCount()

def result(d, p):
    sc = sum_c(d, p)
    if game_code == 0:
        if sc[0] > sc[1]:
            return(0)
        elif sc[0] == sc[1]:
            if len(d) == 2 and sc[0] == 21 and not 10 in d:
                return(0)
            else:
                return(2)
        else:
            return(1)
    else:
        if game_code == -1:
            return(1)
        elif game_code == 1:
            return(0)
        elif game_code == 3:
            return(3)
        elif game_code == 2:
            return(4)
        else:
            return(2)
    
    '''
    0 Dealer Win
    1 Player WIn
    2 Push
    3 Player BlackJack
    4 Surrender
    '''

def line(n):
    if n == 1:
        print('\n----------------\n')
    else:
        print('----------------')


def counting(c):
    global count
    
    if cm == 1:
        if c in P_CARD or c == 10:
            count += -1
        elif c <= 6:
            count += 1
        elif c == 9:
            count += -0.5
        elif c == 7:
            count += 0.5
            
def printCount():
    if cm == 1:
        print('counting: ' + str(count))

def blackJack():
    dc = []
    pc = []
    
    turn = 0
    
    global bet
    global money
    global game_code
    global hidden
    global game_code
    global win
    global lose
    global blj
    global ave_bet
    
    hidden = 1
    
    game_code = 0
    
    if money <= 0:
        print('\n\n----------------\n')
        print('no money left')
        report()
        sys.exit()
        
    printCount()
    
    print('leftCard: ' + str(len(c_arr) - c_s))
    
    print('Money: ' + str(money))
    print('Bet Money -> ', end='')
    bet = int(input())
    if bet >= money:
        print('----- All In -----')
        bet = money
    money -= bet
    
    if ave_bet == 0:
        ave_bet = bet
    else:
        ave_bet = (ave_bet + bet)/2

    line(2)

    pc.append(pick_c())
    dc.append(pick_c())
    dc.append(pick_c())
    pc.append(pick_c())
    dp(dc)
    dp(pc)
    print_c(dc, pc)
    
    
    if sum_c(dc, pc)[1] == 21:
        if not 10 in pc:
            if not sum_c(dc, pc)[0] == 21:
                game_code = 3
            else:
                game_code = 4
    
    ans = -1
    while(ans != 0 and game_code == 0):
        if sum_c(dc, pc)[1] > 21:
            game_code = 1
            print('Player Bust')
            break
        
        turn += 1

        line(2)
        
        print('Hit: 1\nStand: 0')
        if turn == 1:
            print('Surrender: 2\nDouble Down: 3',end=' ')
        print('->',end=' ')
        ans = int(input())
        
        if ans == 1:
            pc.append(pick_c())
        elif ans == 2 and turn == 1:
            game_code = 2
        elif ans == 3 and turn == 1:
            print('----- Double -----')
            bet *= 2
            if bet >= money:
                print('----- All In -----')
                bet = money
            money -= bet
            pc.append(pick_c())
        
        line(2)
        print_c(dc, pc)

    hidden = 0
    
    if game_code == 0:
        while(True):
            if sum_c(dc, pc)[0] >= D_STAND:
                if sum_c(dc, pc)[0] > 21:
                    game_code = -1
                    print('Dealer Bust')
                break
            else:
                dc.append(pick_c())
                
            print_c(dc, pc)
            
    line(1)
    print_c(dc, pc)
    
    res = result(dc, pc)
    if res == 0:
        print('Dealer Win')
        lose += 1
    elif res == 1:
        print('Player Win')
        money += bet * 2
        win += 1
    elif res == 2:
        print('Push')
        money += bet
    elif res == 3:
        print('BlackJack')
        money += math.floor(bet * 2.5)
        win += 1
        blj += 1
    else:
        print('Surrender')
        money += math.floor(bet / 2)


def report():
    line(1)
    print('count: ' + str(cm))
    print('win/lose/all: ' + str(win) + ' / ' + str(lose) + ' / ' + str(m_game))
    print('win rate: ' + str(win/m_game*100))
    print('blackJack: ' + str(blj))
    print('blackjack rate: ' + str(blj/m_game*100))
    print('average bet: ' + str(ave_bet))
    print('money: ' + str(money))
    print('income: ' + str(money - seed))
    print('income rate: ' + str((money-seed)/seed*100))
    
    f.write(','.join([str(cm), str(win), str(lose), str(m_game), str(win/m_game*100), str(blj), str(blj/m_game*100), str(ave_bet), str(money), str(money - seed), str((money-seed)/seed*100)]))
    f.write('\n')
    f.close()


def main():
    global cm
    cm = int(input('count?'))
    
    shuffle()
    
    global game
    print(c_s)
    
    for g in range(m_game):
        line(1)
        print(str(game) + ' game')
        blackJack()
        game += 1
        #print(len(c_arr))
        
    report()


if __name__== "__main__":
    main()

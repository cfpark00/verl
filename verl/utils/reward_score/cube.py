import json
from itertools import groupby

def U(ts,bs):
    # before:
    # ts=[[1, ['W', 'R', 'G']], [2, ['W', 'O', 'G']], [3, ['W', 'O', 'B']], [4, ['W', 'R', 'B']]]
    # bs=[[5, ['Y', 'R', 'G']], [6, ['Y', 'O', 'G']], [7, ['Y', 'O', 'B']], [8, ['Y', 'R', 'B']]]
    # after:
    # ts=[[2, ['O', 'W', 'G']], [6, ['O', 'Y', 'G']], [3, ['W', 'O', 'B']], [4, ['W', 'R', 'B']]]
    # bs=[[1, ['R', 'W', 'G']], [5, ['R', 'Y', 'G']], [7, ['Y', 'O', 'B']], [8, ['Y', 'R', 'B']]]
    ts[0],ts[1],bs[0],bs[1]=ts[1],bs[1],ts[0],bs[0]
    bs[0][1][0],bs[0][1][1]=bs[0][1][1],bs[0][1][0]
    ts[0][1][0],ts[0][1][1]=ts[0][1][1],ts[0][1][0]
    bs[1][1][0],bs[1][1][1]=bs[1][1][1],bs[1][1][0]
    ts[1][1][0],ts[1][1][1]=ts[1][1][1],ts[1][1][0]
def U_(ts,bs):
    for i in range(3):
        U(ts,bs)
def U2(ts,bs):
    for i in range(2):
        U(ts,bs)

def D(ts,bs):
    # before:
    # ts=[[1, ['W', 'R', 'G']], [2, ['W', 'O', 'G']], [3, ['W', 'O', 'B']], [4, ['W', 'R', 'B']]]
    # bs=[[5, ['Y', 'R', 'G']], [6, ['Y', 'O', 'G']], [7, ['Y', 'O', 'B']], [8, ['Y', 'R', 'B']]]
    # after:
    # ts=[[1, ['W', 'R', 'G']], [2, ['W', 'O', 'G']], [4, ['W', 'R', 'B']], [8, ['Y', 'R', 'B']]]
    # bs=[[5, ['Y', 'R', 'G']], [6, ['Y', 'O', 'G']], [3, ['W', 'O', 'B']], [7, ['Y', 'O', 'B']]]
    ts[2],ts[3],bs[2],bs[3]=ts[3],bs[3],ts[2],bs[2]
    ts[2][1][0],ts[2][1][1]=ts[2][1][1],ts[2][1][0]
    ts[3][1][0],ts[3][1][1]=ts[3][1][1],ts[3][1][0]
    bs[2][1][0],bs[2][1][1]=bs[2][1][1],bs[2][1][0]
    bs[3][1][0],bs[3][1][1]=bs[3][1][1],bs[3][1][0]
def D_(ts,bs):
    for i in range(3):
        D(ts,bs)
def D2(ts,bs):
    for i in range(2):
        D(ts,bs)

def R(ts,bs):
    # before:
    # ts=[[1, ['W', 'R', 'G']], [2, ['W', 'O', 'G']], [3, ['W', 'O', 'B']], [4, ['W', 'R', 'B']]]
    # bs=[[5, ['Y', 'R', 'G']], [6, ['Y', 'O', 'G']], [7, ['Y', 'O', 'B']], [8, ['Y', 'R', 'B']]]
    # after:
    # ts=[[1, ['W', 'R', 'G']], [3, ['W', 'O', 'B']], [7, ['Y', 'O', 'B']], [4, ['W', 'R', 'B']]]
    # bs=[[5, ['Y', 'R', 'G']], [2, ['W', 'O', 'G']], [6, ['Y', 'O', 'G']], [8, ['Y', 'R', 'B']]]
    ts[1],ts[2],bs[1],bs[2]=ts[2],bs[2],ts[1],bs[1]
    ts[1][1].reverse()
    ts[2][1].reverse()
    bs[1][1].reverse()
    bs[2][1].reverse()
def R_(ts,bs):
    for i in range(3):
        R(ts,bs)
def R2(ts,bs):
    for i in range(2):
        R(ts,bs)

def L(ts,bs):
    # before:
    # ts=[[1, ['W', 'R', 'G']], [2, ['W', 'O', 'G']], [3, ['W', 'O', 'B']], [4, ['W', 'R', 'B']]]
    # bs=[[5, ['Y', 'R', 'G']], [6, ['Y', 'O', 'G']], [7, ['Y', 'O', 'B']], [8, ['Y', 'R', 'B']]]
    # after:
    # ts=[[5, ['Y', 'R', 'G']], [2, ['W', 'O', 'G']], [3, ['W', 'O', 'B']], [1, ['W', 'R', 'G']]]
    # bs=[[8, ['Y', 'R', 'B']], [6, ['Y', 'O', 'G']], [7, ['Y', 'O', 'B']], [4, ['W', 'R', 'B']]]
    ts[0],ts[3],bs[0],bs[3]=bs[0],ts[0],bs[3],ts[3]
    ts[0][1].reverse()
    ts[3][1].reverse()
    bs[0][1].reverse()
    bs[3][1].reverse()
def L_(ts,bs):
    for i in range(3):
        L(ts,bs)
def L2(ts,bs):
    for i in range(2):
        L(ts,bs)

def F_(ts,bs):
    ts.append(ts.pop(0))
    for i in range(4):
        ts[i][1][1],ts[i][1][2]=ts[i][1][2],ts[i][1][1]
def F(ts,bs):
    for i in range(3):
        F_(ts,bs)
def F2(ts,bs):
    for i in range(2):
        F_(ts,bs)

def B(ts,bs):
    bs.append(bs.pop(0))
    for i in range(4):
        bs[i][1][1],bs[i][1][2]=bs[i][1][2],bs[i][1][1]
def B_(ts,bs):
    # bs.insert(0,bs.pop())
    # for i in range(4):
    #     bs[i][1][1],bs[i][1][2]=bs[i][1][2],bs[i][1][1]
    for i in range(3):
        B(ts,bs)
def B2(ts,bs):
    # bs[0],bs[2]=bs[2],bs[0]
    # bs[1],bs[3]=bs[3],bs[1]
    for i in range(2):
        B(ts,bs)

def get_state(ts,bs):
    return bs[0][1][2]+bs[1][1][2]+ts[0][1][2]+ts[1][1][2]+ts[1][1][1]+bs[1][1][1]+ts[2][1][1]+bs[2][1][1]+ \
        ts[0][1][0]+ts[1][1][0]+ts[3][1][0]+ts[2][1][0]+ts[3][1][2]+ts[2][1][2]+bs[3][1][2]+bs[2][1][2]+ \
        bs[0][1][1]+ts[0][1][1]+bs[3][1][1]+ts[3][1][1]+bs[1][1][0]+bs[0][1][0]+bs[2][1][0]+bs[3][1][0]

def get_tsbs(ts, bs, state):
    bs[0][1][2],bs[1][1][2],ts[0][1][2],ts[1][1][2],ts[1][1][1],bs[1][1][1],ts[2][1][1],bs[2][1][1], \
        ts[0][1][0],ts[1][1][0],ts[3][1][0],ts[2][1][0],ts[3][1][2],ts[2][1][2],bs[3][1][2],bs[2][1][2], \
        bs[0][1][1],ts[0][1][1],bs[3][1][1],ts[3][1][1],bs[1][1][0],bs[0][1][0],bs[2][1][0],bs[3][1][0] = state
    
def apply_move(state, move):
    assert move in ['R', 'RR', 'RRR', 'U', 'UU', 'UUU', 'F', 'FF', 'FFF']
    ts=[[1, ['F', 'L', 'U']], [2, ['F', 'R', 'U']], [3, ['F', 'R', 'D']], [4, ['F', 'L', 'D']]]
    bs=[[5, ['B', 'L', 'U']], [6, ['B', 'R', 'U']], [7, ['B', 'R', 'D']], [8, ['B', 'L', 'D']]]
    get_tsbs(ts, bs, state)
    if move=='R': R(ts,bs)
    elif move=='U': U(ts,bs)
    elif move=='F': F(ts,bs)
    elif move=='RR': 
        R(ts,bs)
        R(ts,bs)
    elif move=='UU': 
        U(ts,bs)
        U(ts,bs)
    elif move=='FF': 
        F(ts,bs)
        F(ts,bs)
    elif move=='RRR': 
        R(ts,bs)
        R(ts,bs)
        R(ts,bs)
    elif move=='UUU': 
        U(ts,bs)
        U(ts,bs)
        U(ts,bs)
    elif move=='FFF': 
        F(ts,bs)
        F(ts,bs)
        F(ts,bs)
    return get_state(ts,bs)

def apply_moves(state, moves):
    moves=''.join(moves)
    ts=[[1, ['F', 'L', 'U']], [2, ['F', 'R', 'U']], [3, ['F', 'R', 'D']], [4, ['F', 'L', 'D']]]
    bs=[[5, ['B', 'L', 'U']], [6, ['B', 'R', 'U']], [7, ['B', 'R', 'D']], [8, ['B', 'L', 'D']]]
    get_tsbs(ts, bs, state)
    for move in moves:
        assert move in ['R', 'U', 'F'], f"Invalid move: {move}"
        if move=='R': R(ts,bs)
        elif move=='U': U(ts,bs)
        elif move=='F': F(ts,bs)
    return get_state(ts,bs)

def solved_prompt(prompt):
    state = apply_moves(prompt[:24], [''.join(g) for _, g in groupby(prompt[24:])])
    if state.lower() == "uuuurrrrffffddddllllbbbb":
        return True
    else:
        return False

def to_words(text):
    if not text:
        return text
    result = text[0]
    for i in range(1, len(text)):
        # If both the current and previous characters are uppercase and the same, no space is added.
        if text[i].isupper() and text[i-1].isupper() and text[i] == text[i-1]:
            result += text[i]
        else:
            result += " " + text[i]
    return result

def to_text(words):
    return "".join(words).replace(" ", "")

def compute_score(solution_str, ground_truth):
    assert ground_truth=="", "ground_truth should be empty for cube"
    #print("solution_str",solution_str)
    solution_str = solution_str.replace("<|endoftext|>", "").strip()
    text=to_text(solution_str)
    try:
        solved=solved_prompt(text)
    except:
        return 0.0
    #print("solved",solved)
    if solved:
        return 1.0
    else:
        return 0.0
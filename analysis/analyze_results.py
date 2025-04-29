import sys, os
import json
import numpy as np
import matplotlib.pyplot as plt

# read json file

result_dir = '/n/netscratch/dam_lab/Everyone/wall/cfpark00/baselines/'
expts = [
    'qwen-2.5-1.5b-instruct',
    'qwen-2.5-1.5b-instruct-grpo-math-v1-130',
    # 'qwen-2.5-1.5b-instruct-grpo-gsm8k-v1-7',
    # 'qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-130'
    ]
file_path = 'math_500/temp=1.0_seed=none/data.json'


plt.figure()
correct_probs_expt = []
for e in expts:
    file_name = os.path.join(result_dir, e, file_path)
    with open(file_name, 'r') as f:
        data = json.load(f)

    # keys include: 'id', 'gt_answer', 'model_answers', 'votes_sympy', 'correct_vote_sympy', 'corrects_sympy'

    correct_probs = []
    for d in data:
        probs = np.mean(d['corrects_sympy'])
        correct_probs.append(probs)
    correct_probs = np.array(correct_probs)

    if e == 'qwen-2.5-1.5b-instruct':
        difficulty_idx = np.argsort(correct_probs)
    
    # plt.plot(np.arange(len(correct_probs)), correct_probs[difficulty_idx], label=e, ls='.')
    correct_probs_expt.append(correct_probs)
    
# plt.scatter(correct_probs_expt[0], correct_probs_expt[1])
plt.hexbin(correct_probs_expt[0], correct_probs_expt[1], gridsize=50, cmap="plasma", bins='log') 
# plot x = y
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--')
# Add colorbar
plt.colorbar(label="Density")
plt.xlabel('Base')
plt.ylabel('PPM GSM8k Trained')
plt.savefig('correct_ppo.png')
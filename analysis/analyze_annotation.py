import json 
import numpy as np
import matplotlib.pyplot as plt

base_annotation_file = '/n/home05/sqin/wall/verl/analysis/gpt4o_annotate_qwen1.5b.json'
grpo_annotation_file = '/n/home05/sqin/wall/verl/analysis/gpt4o_annotate_qwen1.5b_gropo_math.json'


# read json file
with open(base_annotation_file, 'r') as f:
    base_data = json.load(f)

with open(grpo_annotation_file, 'r') as f:
    grpo_data = json.load(f)

def get_correct_probs(data):
    correct_probs = []
    orig_probs = [] 
    for quest_id, val in data.items():
        response_annot = val['response_annotation']
        orig_correct = 32 - len(response_annot)
        annot_correct = np.sum([1 for r in response_annot if ': Correct' in r or ': Formatting error' in r])
        correct_probs.append((orig_correct + annot_correct) / 32)
        orig_probs.append(orig_correct / 32)
    correct_probs = np.array(correct_probs)
    orig_probs = np.array(orig_probs)
    return correct_probs, orig_probs

base_correct_probs, base_orig_probs = get_correct_probs(base_data)
grpo_correct_probs, grpo_orig_probs = get_correct_probs(grpo_data)

# pad array to length 500 with 1s
base_correct_probs = np.pad(base_correct_probs, (0, 500 - len(base_correct_probs)), 'constant', constant_values=(1))
grpo_correct_probs = np.pad(grpo_correct_probs, (0, 500 - len(grpo_correct_probs)), 'constant', constant_values=(1))
base_orig_probs = np.pad(base_orig_probs, (0, 500 - len(base_orig_probs)), 'constant', constant_values=(1))
grpo_orig_probs = np.pad(grpo_orig_probs, (0, 500 - len(grpo_orig_probs)), 'constant', constant_values=(1))

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(base_correct_probs)), np.sort(base_correct_probs)[::-1], label='Qwen2.5-1.5B-Instruct: Graded by Sympy', color='blue')  
plt.plot(np.arange(len(base_orig_probs)), np.sort(base_orig_probs)[::-1], label='Qwen2.5-1.5B-Instruct: Graded by GPT-4o', ls='--', color='blue')

plt.plot(np.arange(len(grpo_correct_probs)), np.sort(grpo_correct_probs)[::-1], label='Qwen2.5-1.5B-Instruct-GRPO-MATH: Graded by Sympy', color='red')
plt.plot(np.arange(len(grpo_orig_probs)), np.sort(grpo_orig_probs)[::-1], label='Qwen2.5-1.5B-Instruct-GRPO-MATH: Graded by GPT-4o', ls='--', color='red')


plt.legend()
plt.xlabel('Question Rank')
plt.ylabel('Correctness Probs (Pass@32)')
plt.savefig('correct_probs_annot.png')

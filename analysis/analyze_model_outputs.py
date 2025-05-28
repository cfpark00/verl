import sys, os
import json
import numpy as np
from openai import OpenAI

client = OpenAI(
)

def format_question(message, gt_answer):
    quest = message['content']
    formatted_question = f'''
    Question: {quest}
    Correct Answer: {gt_answer}
    '''
    return formatted_question


def get_questions_to_annotate(outputs, correctness, boxed_answer):
    questions_to_annotate = {}
    for quest, ans in outputs.items():
        orig_correct = correctness[quest]
        # # find instances where the original model gets wrong
        incorrect_attempts = []

        for i in range(len(orig_correct)):
            if orig_correct[i] < 1:
                incorrect_attempts.append(
                    (ans[i], boxed_answer[quest][i])
                    )
        if len(incorrect_attempts) > 0:
            questions_to_annotate[quest] = incorrect_attempts
    return questions_to_annotate
        
    
def ask_openai(formatted_question, attempted_solution):
    input_text = f'''
        Below is a math problem and the correct answer. 
        {formatted_question}
        Below is an attempt to solve the problem.
        Please carefully read the attempted solution and annotate whether each of the solution is correct. 
        If the solution is correct, please only output "Correct".
        If the solution is incorrect, but is incorrect only due to incorrect formatting (for example, not simplifying the expression or expressing in a different format), please output "Formatting error".
        Otherwise, please give respond with "Incorrect" and give a one sentence explanation of why the solution is incorrect.
        Attempted Solution:
        {attempted_solution}

    '''
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": input_text
            }
        ]
    )
    return completion



# get model results: 
result_dir = '/n/netscratch/dam_lab/Everyone/wall/cfpark00/baselines/'
expts = [
    'qwen-2.5-1.5b-instruct',
    'qwen-2.5-1.5b-instruct-grpo-math-v1-130',
    # 'qwen-2.5-1.5b-instruct-grpo-gsm8k-v1-7',
    # 'qwen-2.5-1.5b-instruct-ppo-gsm8k-v1-130'
    ]
file_path = 'math_500/temp=1.0_seed=none/data.json'

orginal_correctness = {}
original_outputs = {}
original_boxed_answer = {}

post_rl_correctness = {}
post_rl_outputs = {}
post_rl_boxed_answer = {}

question = {}
for e in expts:
    file_name = os.path.join(result_dir, e, file_path)
    with open(file_name, 'r') as f:
        data = json.load(f)

    # keys include: ['id', 'gt_answer', 'messages', 'model_answers', 'responses', 'corrects_sympy', 'corrects_gpt', 'corrects_sympy2', 'votes_sympy', 'correct_vote_sympy']
    for i in range(len(data)):
        quest = data[i]['id']
        print(data[i]['messages'][0])
        question[quest] = format_question(data[i]['messages'][0], data[i]['gt_answer'])

        if e == 'qwen-2.5-1.5b-instruct':
            orginal_correctness[quest] = np.array(data[i]['corrects_sympy'])
            original_outputs[quest] = data[i]['responses']
            original_boxed_answer[quest] = data[i]['model_answers']

        else:
            post_rl_correctness[quest] = np.array(data[i]['corrects_sympy'])
            post_rl_outputs[quest] = data[i]['responses']
            post_rl_boxed_answer[quest] = data[i]['model_answers']


def annotate(questions_to_annotate):
    openai_responses = {}
    for quest_id in questions_to_annotate.keys():
        print("input question", question[quest_id])
        response_list = []
        for i in range(len(questions_to_annotate[quest_id])): 
            full_response = questions_to_annotate[quest_id][i][0]
            boxed_ans = questions_to_annotate[quest_id][i][1]
            print("input\n", full_response)
            completion = ask_openai(question[quest_id], full_response)
            print("output\n", completion.choices[0].message.content)
            print("----------------------------------------------------")
            response_list.append(f"{boxed_ans}: {completion.choices[0].message.content}")
        openai_responses[quest_id] = {
                "question": question[quest_id],
                "response_annotation": response_list,
            }
        
    return openai_responses 

if __name__ == "__main__":
    # annotate original model outputs
    questions_to_annotate = get_questions_to_annotate(
        outputs=original_outputs,
        correctness=orginal_correctness,
        boxed_answer=original_boxed_answer
    )
    openai_responses = annotate(questions_to_annotate)
    with open('gpt4o_annotate_qwen1.5b.json', 'w') as f:
        json.dump(openai_responses, f, indent=4)

    # # annotate post-RL model outputs
    # questions_to_annotate = get_questions_to_annotate(
    #     outputs=post_rl_outputs,
    #     correctness=post_rl_correctness,
    #     boxed_answer=post_rl_boxed_answer
    # )
    # openai_responses = annotate(questions_to_annotate)
    # with open('gpt4o_annotate_qwen1.5b_gropo_math.json', 'w') as f:
    #     json.dump(openai_responses, f, indent=4)


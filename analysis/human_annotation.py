import argparse
import os, sys
import json
import re 
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tiktoken


# client = OpenAI()

def load_model_response(model_size, result_file):
    reponse_dir = f'/n/netscratch/dam_lab/Everyone/wall/cfpark00/for_manual_inspection/{model_size}'
    result_path = os.path.join(reponse_dir, result_file)
    # load json file
    with open(result_path, 'r') as f:  
        data = json.load(f)
    print(data[0].keys())
    # data contains a list of dictionaries for each problem 
    # dict keys: 'problem', 'solution', 'answer', 'subject', 'level', 'id', 'messages', 'prompt', 'responses', 'corrects_verl_batched_responses', 'verl_model_answers'
    return data

def extract_options(text):
    """
    Extracts two option groups from a substring like "[YES/NO, YES/NO/N.A.]"
    Returns a tuple of two lists.
    """
    pattern = r"\[\**(YES|NO),\s*(YES|NO|N\.A\.)\**\]"
    match = re.search(pattern, text)
    converting_dict = {'YES': 1, 'NO': 0, 'N.A.': -1}

    if match:
        return converting_dict[match.group(1)], converting_dict[match.group(2)]
    else:
        return None, None

def gpt_idea_summary(args):
    # load the model response
    data = load_model_response(args.model_size, args.result_file)

    # manually inspect 500 test problems
    gt_idea_summary_all = {}
    for i in tqdm(range(len(data))):
    # for i in range(5):
        unique_id = data[i]['id']
        problem = data[i]['problem']
        gt_solution = data[i]['solution']

        instruction = f"Below is a math problem and it's solution trace: \n" \
            f"Problem: {problem} \n" \
            "Solution: " + str(gt_solution) + "\n" \
            f"Please use a few sentences to briefly describe the major steps required to solve this problem. "\
            f"Please do not include any mathematical details of the solution. "\
            f"The general steps should only outline the key steps required to correctly solve the problem. "\
            f"If the solution require consider different cases, please list these cases as well. Otherwise, no need to mention separate cases. "\
            f"Please start the reponse with: Here are the major steps required to solve this type of problem: \n"
            # old prompt
            # f"Please use a few sentences to briefly describe the general idea of how to approach this type of problems. " \
            # f"Please do not include any mathematical details of the solution. " \
            # f"The general idea should apply to any problems following a similar format (i.e., by changing numbers or wordings). "\

        response = client.responses.create(
            model="gpt-4.1",
            input=instruction,
            temperature=0,
            # service_tier="flex",
        )

        idea_summary = response.output_text
        print("Idea Summary: ", idea_summary)
        gt_idea_summary_all[unique_id] = idea_summary
    
    # save results as json file
    result_dir = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}'
    gt_summary_file = os.path.join(result_dir, f'gt_idea_summary.json')
    with open(gt_summary_file, 'w') as f:
        json.dump(gt_idea_summary_all, f)
    print("Results saved to: ", gt_summary_file)
    
    return

def batch_request_json_creation(args):
    # load the model response
    data = load_model_response(args.model_size, args.result_file)
    # load idea summary
    gt_idea_summary_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b/gt_idea_summary.json' 
    with open(gt_idea_summary_file, 'r') as f:
        gt_idea_summary_all = json.load(f)

    # create batch request to manually inspect 500 test problems
    all_requests = []
    for i in tqdm(range(len(data))):
        unique_id = data[i]['id']
        problem = data[i]['problem']
        gt_solution = data[i]['solution']
        gt_idea_summary = gt_idea_summary_all[unique_id]

        # grade student's solution
        grading_requests = []
        for j in range(len(data[i]['responses'])):
            grading_instruction = '''Now, please carefully read through the following solution and answer two questions below. \n 
            (1) Does the solution contain attempts to solve the problem using the approach described above? At this point, it is okay if the solution makes mathematical errors as long as the approach itself is correct. \n 
            (2) If the previous answer is yes, does the solution correctly execute required solution steps without making any critical math errors? Please use "N.A." if the answer to the previous question is "No". \n
            Feel free to elaborate but please provide a final answer using format: [YES/NO, YES/NO/N.A.].'''

            student_solution = data[i]['responses'][j]
            # print("j", "Student Solution: ", student_solution)
            
            instruction = f"You are a responsible grader and your task is to grade a competition-level high school math exam. \n" \
                "Below is a problem, it's solution trace and the general idea behind the solution: \n" \
                f"Problem: {problem} \n" \
                "Solution Trace: " + str(gt_solution) + "\n" \
                "Solution Idea: " + str(gt_idea_summary) + "\n" \
                + str(grading_instruction) + "\n" \
                + "Studednt Solution Attempt: " + str(student_solution) + "\n" 

            # create dict entry file for batch request
            request_entry = {
                "custom_id": f"{args.model_size}_{args.result_file}-{i}-{j}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-4.1",
                    "input": instruction,
                    "temperature": 0,
                    }
            }
            grading_requests.append(request_entry)
            
            all_requests.append(request_entry)
    
    # now dump the requests to the file
    grading_request_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}/{args.result_file}_new_grading_request.jsonl'
    with open(grading_request_file, "w") as f:
        for obj in all_requests:
            f.write(json.dumps(obj) + "\n")

    return

def upload_batch_request(args):
    grading_request_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/{args.model_size}/{args.result_file}_new_grading_request.jsonl'
    print("Uploading grading request file: ", grading_request_file)
    batch_input_file = client.files.create(
        file=open(grading_request_file, "rb"),
        purpose="batch"
    )

    print(batch_input_file)
    return 

def submit_batch_request(batch_input_file_id):
    print("Submitting grading request file")
    # batch_input_file_id = batch_input_file.id
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "description": "Grading Request",
        }
    )
    print("Batch request submitted.")
    print(batch_info)
    return

def check_status(batch_id):
    # Check the status of the batch
    batch_info = client.batches.retrieve(batch_id)
    print("Batch Status: ", batch_info)
    return

def retrieve_results(batch_id):
    # Retrieve the results of the batch
    batch_results = client.files.content(batch_id)
    # print("Batch Results: ", batch_results.text)
    # save as a jsonl file
    result_dir = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/7b'
    result_file = f'post_temp=1.0_n=64.json_graded_new.json'
    file_path = os.path.join(result_dir, result_file)
    with open(file_path, 'w') as f:
        f.write(batch_results.text)
    print("Results saved to: ", file_path)
    return

def summarize_results_1_5b():
    result_dir = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b'
    graded_files = ['pre_temp=1.0_n=64.json_graded_new.json', 'post_temp=1.0_n=64.json_graded_new.json']
    grading_response_files = []
    direction_grades_files = []
    execution_grades_files = []
    for file in graded_files:
        file_path = os.path.join(result_dir, file)
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        print(file, "Number of graded responses: ", len(data))
        grading_response_all = {}
        # for graded in data:
        for i in range(15304): # partial results 
            graded = data[i]
            grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]
            print("grading_response: ", grading_response)
            enc = tiktoken.encoding_for_model("gpt-4")
            print(len(enc.encode(grading_response)))
            quit()
            grading_id = graded["custom_id"]
            id= re.search(r"\.json-(\d+)-(\d+)$", grading_id)
            problem_id, response_id = int(id.group(1)), id.group(2)
            direction_grade, execution_grade = extract_options(grading_response)
            if direction_grade is None or execution_grade is None:
                continue
            if problem_id not in grading_response_all:
                grading_response_all[problem_id] = [[direction_grade, execution_grade]]
            else:
                grading_response_all[problem_id].append([direction_grade, execution_grade])
        grading_response_files.append(grading_response_all)
        
        avg_direction_grades_all = np.empty(500)
        avg_execution_grades_all = np.empty(500)
        avg_direction_grades_all[:] = np.nan
        avg_execution_grades_all[:] = np.nan
        for key, val in grading_response_all.items():
            # calculate the average of the two grades
            direction_grades = [x[0] for x in val]
            execution_grades = [x[1] for x in val if x[1] != -1]
            avg_direction_grades = sum(direction_grades) / len(direction_grades)
            avg_direction_grades_all[key] = avg_direction_grades
            if len(execution_grades) == 0:
                avg_execution_grades = np.nan
            else:
                avg_execution_grades = sum(execution_grades) / len(execution_grades)
            avg_execution_grades_all[key] = avg_execution_grades

        avg_direction_grades_all = np.array(avg_direction_grades_all)
        avg_execution_grades_all = np.array(avg_execution_grades_all)  

        direction_grades_files.append(avg_direction_grades_all)
        execution_grades_files.append(avg_execution_grades_all)

    direction_grades_files = np.array(direction_grades_files)
    execution_grades_files = np.array(execution_grades_files)
    # save as npy file
    np.savez('1_5b_execution_vs_direction_grades.npy', 
             direction_grades_files=direction_grades_files, 
             execution_grades_files=execution_grades_files)
    
    plt.figure()
    sort_idx = np.argsort(direction_grades_files[0])[::-1]
    plt.scatter(list(range(len(sort_idx))), direction_grades_files[0][sort_idx], label='Pre-GRPO')
    # sort_idx = np.argsort(direction_grades_files[1])[::-1]
    plt.scatter(list(range(len(sort_idx))), direction_grades_files[1][sort_idx], label='Post-GRPO')
    plt.xlabel('Problem ID')
    plt.ylabel('Average Direction Grade')
    plt.title('Average Direction Grades for Pre and Post GRPO')
    plt.legend()
    plt.savefig('1.5b_direction_grades_new.png')


    plt.figure()
    sort_idx = np.argsort(execution_grades_files[0])[::-1]
    plt.scatter(list(range(500)), execution_grades_files[0][sort_idx], label='Pre-GRPO')
    plt.scatter(list(range(500)), execution_grades_files[1][sort_idx], label='Post-GRPO')
    plt.xlabel('Problem ID')
    plt.ylabel('Average Execution Grade')
    plt.title('Average Execution Grades for Pre and Post GRPO')
    plt.legend()
    plt.savefig('1.5b_execution_grades_new.png')
    
    return

def summarize_results_7b():
    result_dir = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/7b'
    graded_files = ['pre_temp=1.0_n=64.json_graded_new.json', 'post_temp=1.0_n=64.json_graded_new.json']
    grading_response_files = []
    direction_grades_files = []
    execution_grades_files = []
    for file in graded_files:
        file_path = os.path.join(result_dir, file)
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        print(file, "Number of graded responses: ", len(data))
        grading_response_all = {}
        # for graded in data:
        for i in range(7632): # partial results 
            graded = data[i]
            grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]
            grading_id = graded["custom_id"]
            id= re.search(r"\.json-(\d+)-(\d+)$", grading_id)
            problem_id, response_id = int(id.group(1)), id.group(2)
            direction_grade, execution_grade = extract_options(grading_response)
            if direction_grade is None or execution_grade is None:
                continue
            if problem_id not in grading_response_all:
                grading_response_all[problem_id] = [[direction_grade, execution_grade]]
            else:
                grading_response_all[problem_id].append([direction_grade, execution_grade])
        grading_response_files.append(grading_response_all)
        
        avg_direction_grades_all = np.empty(500)
        avg_execution_grades_all = np.empty(500)
        avg_direction_grades_all[:] = np.nan
        avg_execution_grades_all[:] = np.nan
        for key, val in grading_response_all.items():
            # calculate the average of the two grades
            direction_grades = [x[0] for x in val]
            execution_grades = [x[1] for x in val if x[1] != -1]
            avg_direction_grades = sum(direction_grades) / len(direction_grades)
            avg_direction_grades_all[key] = avg_direction_grades
            if len(execution_grades) == 0:
                avg_execution_grades = np.nan
            else:
                avg_execution_grades = sum(execution_grades) / len(execution_grades)
            avg_execution_grades_all[key] = avg_execution_grades

        avg_direction_grades_all = np.array(avg_direction_grades_all)
        avg_execution_grades_all = np.array(avg_execution_grades_all)  

        direction_grades_files.append(avg_direction_grades_all)
        execution_grades_files.append(avg_execution_grades_all)
    
    direction_grades_files = np.array(direction_grades_files)
    execution_grades_files = np.array(execution_grades_files)
    # save as npy file
    np.savez('7b_execution_vs_direction_grades.npy', 
             direction_grades_files=direction_grades_files, 
             execution_grades_files=execution_grades_files)
    
    plt.figure()
    sort_idx = np.argsort(direction_grades_files[0])[::-1]
    plt.scatter(list(range(len(sort_idx))), direction_grades_files[0][sort_idx], label='Pre-GRPO')
    sort_idx = np.argsort(direction_grades_files[1])[::-1]
    plt.scatter(list(range(len(sort_idx))), direction_grades_files[1][sort_idx], label='Post-GRPO')
    plt.xlabel('Problem ID')
    plt.ylabel('Average Direction Grade')
    plt.title('Average Direction Grades for Pre and Post GRPO')
    plt.legend()
    plt.savefig('7b_direction_grades_new.png')


    plt.figure()
    sort_idx = np.argsort(execution_grades_files[0])[::-1]
    plt.scatter(list(range(500)), execution_grades_files[0][sort_idx], label='Pre-GRPO')
    plt.scatter(list(range(500)), execution_grades_files[1][sort_idx], label='Post-GRPO')
    plt.xlabel('Problem ID')
    plt.ylabel('Average Execution Grade')
    plt.title('Average Execution Grades for Pre and Post GRPO')
    plt.legend()
    plt.savefig('7b_execution_grades_new.png')
    
    return

def compare_two_models():
    results_1_5b = np.load('1_5b_execution_vs_direction_grades.npz')
    results_7b = np.load('7b_execution_vs_direction_grades.npz')
    direction_grades_1_5b = results_1_5b['direction_grades_files']
    direction_grades_7b = results_7b['direction_grades_files'] 
    execution_grades_1_5b = results_1_5b['execution_grades_files']
    execution_grades_7b = results_7b['execution_grades_files'] 

    # first, show that execution difficulty is not correlated with direction difficulty
    # compute correlation
    mask = ~np.isnan(direction_grades_1_5b[0]) & ~np.isnan(execution_grades_1_5b[0])
    corr = np.corrcoef(direction_grades_1_5b[0][mask], execution_grades_1_5b[0][mask])[0, 1]

    plt.figure(figsize=(6, 4))
    plt.scatter(execution_grades_1_5b[0], direction_grades_1_5b[0], label='Pre-GRPO 1.5B, corr=%.2f' % corr, color='C0')
    plt.ylabel('Pass@k on Solution Direction')
    plt.xlabel('Pass@k on Solution Execution')
    plt.title('Execution Difficulty vs Direction Difficulty')
    plt.legend()
    plt.savefig('1_5b_execution_vs_direction_difficulty.png', bbox_inches='tight')
    

    # default baseline is 1.5b pre GRPO
    sort_idx = np.argsort(direction_grades_1_5b[0])[::-1]

    # plot the two models, four subplots
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    # compare POST GRPO model of the same size
    plt.scatter(list(range(len(sort_idx))), direction_grades_1_5b[0][sort_idx], label='1.5b Pre-GRPO', color='C0')    
    plt.scatter(list(range(len(sort_idx))), direction_grades_1_5b[1][sort_idx], label='1.5 Post-GRPO', color='C2')
    plt.xlabel('Problem ID')
    plt.ylabel('Pass@k on Solution Direction')
    plt.title('Average Direction Grades for 1.5B Pre vs Post GRPO')
    plt.legend()

    # compare PRE GRPO model of different sizes
    plt.subplot(1, 2, 2)
    plt.scatter(list(range(len(sort_idx))), direction_grades_1_5b[0][sort_idx], label='1.5b Pre-GRPO', color='C0')    
    plt.scatter(list(range(len(sort_idx))), direction_grades_7b[0][sort_idx], label='7b Pre-GRPO', color='C1')
    plt.xlabel('Problem ID')
    plt.ylabel('Pass@k on Solution Direction')
    plt.title('Average Direction Grades for Pre GRPO 1.5B vs 7B')
    plt.legend()

    plt.savefig('1_5b_vs_7b_direction_grades_comparison.png', bbox_inches='tight')

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sort_idx = np.argsort(execution_grades_1_5b[0])[::-1]
    # compare POST GRPO model of the same size
    plt.scatter(list(range(len(sort_idx))), execution_grades_1_5b[0][sort_idx], label='1.5b Pre-GRPO', color='C0', marker='x')    
    plt.scatter(list(range(len(sort_idx))), execution_grades_1_5b[1][sort_idx], label='1.5 Post-GRPO', color='C2', marker='x')
    plt.xlabel('Problem ID')
    plt.ylabel('Pass@k on Solution Execution')
    plt.xlim([250, 500])
    plt.title('Average Execution Grades for 1.5B Pre vs Post GRPO')
    plt.legend()

    # compare PRE GRPO model of different sizes
    plt.subplot(1, 2, 2)
    plt.scatter(list(range(len(sort_idx))), execution_grades_1_5b[0][sort_idx], label='1.5b Pre-GRPO', color='C0', marker='x')    
    plt.scatter(list(range(len(sort_idx))), execution_grades_7b[0][sort_idx], label='7b Pre-GRPO', color='C1', marker='x')
    plt.xlabel('Problem ID')
    plt.xlim([250, 500])
    plt.ylabel('Pass@k on Solution Execution')
    plt.title('Average Execution Grades for Pre GRPO 1.5B vs 7B')
    plt.legend()

    plt.savefig('1_5b_vs_7b_execution_grades_comparison.png', bbox_inches='tight')

    return

def examine_coverage_problems():
    results_1_5b = np.load('1_5b_execution_vs_direction_grades.npz')
    direction_grades_1_5b = results_1_5b['direction_grades_files']
    execution_grades_1_5b = results_1_5b['execution_grades_files']
    execution_pre = execution_grades_1_5b[0]
    execution_post = execution_grades_1_5b[1]

    result_dir = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b'
    graded_files = ['pre_temp=1.0_n=64.json_graded_new.json', 'post_temp=1.0_n=64.json_graded_new.json']
    graded_response_files = []
    for file in graded_files:
        file_path = os.path.join(result_dir, file)
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        graded_response_files.append(data)

    for i in range(500):
        if execution_pre[i] == 0 and execution_post[i] > 0:
            print("Problem ID: ", i)
            print("Pre GRPO Execution Grade: ", execution_pre[i])
            print("Post GRPO Execution Grade: ", execution_post[i])
            print("\n")
    # let's examine problem 189
    problem_id = 10
    # first, read the question
    problem_data_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b/pre_temp=1.0_n=64.json_new_grading_request.jsonl'
    # load the jsonl file
    with open(problem_data_file, "r") as f:
        question_data = []
        for line in f:
            question_data.append(json.loads(line))
    for question in question_data:
        if question["custom_id"].startswith(f"1.5b_pre_temp=1.0_n=64.json-{problem_id}-"):
            question = question["body"]["input"]
            print("Problem: ", question)
            break

    # now examine the pre GRPO response
    for graded in graded_response_files[0]:
        if graded["custom_id"].startswith(f"1.5b_pre_temp=1.0_n=64.json-{problem_id}-"):
            grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]
   
            grading_id = graded["custom_id"]
            id= re.search(r"\.json-(\d+)-(\d+)$", grading_id)
            problem_id, response_id = int(id.group(1)), id.group(2)
            direction_grade, execution_grade = extract_options(grading_response)
            
            if execution_grade == 0:
                print("="*100)
                print(f"Pre GRPO Execution Grade {response_id}: ", execution_grade)
                # print("Pre GRPO Grading Response: ", grading_response)
            # print(f"Pre GRPO Direction Grade {response_id}: ", direction_grade)

    # now examine the post GRPO response
    for graded in graded_response_files[1]:
        if graded["custom_id"].startswith(f"1.5b_post_temp=1.0_n=64.json-{problem_id}-"):
            grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]
   
            grading_id = graded["custom_id"]
            id= re.search(r"\.json-(\d+)-(\d+)$", grading_id)
            problem_id, response_id = int(id.group(1)), id.group(2)
            direction_grade, execution_grade = extract_options(grading_response)
            
            if execution_grade > 0:
                print("*"*100)
                print(f"Post GRPO Execution Grade {response_id}: ", execution_grade)
                # print("Post GRPO Grading Response: ", grading_response)
    print("DONE")
    return    

def estimate_api_cost():
    enc = tiktoken.encoding_for_model("gpt-4")

    # average tokens for prompt:
    problem_data_file = f'/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b/pre_temp=1.0_n=64.json_new_grading_request.jsonl'
    # load the jsonl file
    with open(problem_data_file, "r") as f:
        question_data = []
        for line in f:
            question_data.append(json.loads(line))
    prompt_len_count = []
    for question in question_data:
        question = question["body"]["input"]
        prompt_len_count.append(len(enc.encode(question)))
    print("Average tokens for prompt: ", np.mean(prompt_len_count))
    print("Total number of requests: ", len(question_data))

    # get average tokens for response:
    result_dir = '/n/netscratch/dam_lab/Everyone/wall/sunny/for_manual_inspection/1.5b'
    graded_files = ['pre_temp=1.0_n=64.json_graded_new.json']
    for file in graded_files:
        file_path = os.path.join(result_dir, file)
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        print(file, "Number of graded responses: ", len(data))
        resp_len_count = []
        for i in range(15304): 
            graded = data[i]
            grading_response = graded["response"]["body"]["output"][0]["content"][0]["text"]
            
            resp_len_count.append(len(enc.encode(grading_response)))
        print("Average tokens per response: ", np.mean(resp_len_count))
    
    # compute the cost
    # input tokens
    input_tokens = np.mean(prompt_len_count) * len(question_data)
    input_price = input_tokens / 1_000_000 * 0.2 # $1.00 per 1M tokens
    # output tokens
    output_tokens = np.mean(resp_len_count) * len(question_data)
    output_price = output_tokens / 1_000_000 * 0.8 # $4.00 per 1M tokens
    # total cost
    total_cost = input_price + output_price
    print("Total cost: ", total_cost)
    
    return 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Annotation Script")

    parser.add_argument("--model_size", type=str, default="1.5b", 
                        help="1.5b or 7b")
    parser.add_argument("--result_file", type=str, default="pre_temp=1.0_n=64.json", 
                        help="/n/netscratch/dam_lab/Everyone/wall/cfpark00/for_manual_inspection")
    
    args = parser.parse_args()
    
    # gpt_idea_summary(args)

    # batch_request_json_creation(args)

    # upload_batch_request(args)

    # 1.5b 
    # pre_grpo_batch_input_file_id = 'file-GW3HeNemibdimMpNgtUVFQ'
    # post_grpo_batch_input_file_id = 'file-27FZVMRzAiyuGcxyBXTnbR'
    # 7b 
    # pre_grpo_batch_input_file_id = 'file-JmPXq4Ubk19ddF7H4Ve3ZR'
    # post_grpo_batch_input_file_id = 'file-TQTgLMBJ3CdhsJnCXHiE1X'
    # submit_batch_request(batch_input_file_id = post_grpo_batch_input_file_id)

    # pre_grpo_batch_id = 'batch_6806a69761c881908e9bc431d375cafa' # 1.5b - pre GRPO new solution summary
    # post_grpo_batch_id = 'batch_6806a84c008c8190b69885b907f09d11' # 1.5b - post GRPO new solution summary
    # pre_grpo_batch_id = 'batch_6806aa1da10481908539a5a3b29b1b98' # 7b - pre GRPO new solution summary
    # post_grpo_batch_id = 'batch_6806aacc69048190bb9ccb28e368835f' # 7b - post GRPO new solution summary
    # check_status(batch_id=post_grpo_batch_id)

    # 1.5b 
    # pre_grpo_output_file_id = 'file-GRzDWCFjCEgceRjE4LUdZr'
    # post_grpo_output_file_id = 'file-7wM764HtX2UPBPNc7oGkAR'
    # 7b 
    # pre_grpo_output_file_id = 'file-DzrmzBgiqyciGvB53izRqA'
    # post_grpo_output_file_id = 'file-5U1qTC6VDRYPZeFB3ptfRG'
    # retrieve_results(batch_id=post_grpo_output_file_id)

    # summarize_results_1_5b()

    # summarize_results_7b()

    # compare_two_models()

    # examine_coverage_problems()

    estimate_api_cost()
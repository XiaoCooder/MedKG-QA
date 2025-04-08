import structllm as sllm
import json
from tqdm import tqdm
import openai
import ast

def evaluate_answer_quality(args):
    llm = sllm.llm.gpt(args)
    path = "/home/wcy/code/KG-MedQA-v1.0/output/ceshi/llm-deepseek-chat__SentenceBERT__bs-10__20250408_184255/qa_history.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    total_num = 0
    correct = 0
    matched = 0
    answers = []
    p_answers = []
    for item in tqdm(data, desc="评估问答准确率"):
        question = item['Q']
        answer = item['A']
        p_answers.append(answer)
        triples_text = item['used_triples']

        # step 1: 构造模型回答 prompt（用已知 triples 回答问题）
        query_prompt = sllm.query_prompt.query_prompt(args, triples_text)
        query_prompt.create_prompt(task="get_answer1", question=question)
        responses = llm.get_response(query_prompt.naive_prompt)
        for response in responses:
            try:
                result = response.choices[0].message.content
                answer_response = sllm.align.get_answer(result)
                answers.append(answer_response)
                
            except openai.BadRequestError as e: # 非法输入 '$.input' is invalid. query返回结果为：请输入详细信息等
                print(e)
                total_num += 1
                continue

            except IndexError as e: # 得不到正确格式的query: set1=(fastest car)
                print(e)
                total_num += 1 # 防止卡死
                continue

            except openai.APITimeoutError as e: # 超时
                print(e)
                total_num += 1 # 防止卡死
                continue

            except ValueError as e: # maximum context length
                print(e)
                continue

            except Exception as e: # 其他错误
                print(e)
                total_num += 1 # 防止卡死
                continue

    # step 2: 评估生成答案与标准答案是否相似
    check_prompt = sllm.query_prompt.query_prompt(args, p_answers)
    check_prompt.create_prompt(task="judge_acc", question=answers)
    responses = llm.get_response(check_prompt.naive_prompt)
    for response in responses:
        try:
            result_str = response.choices[0].message.content.strip()
            # 把 '[yes,no,yes]' 转换成 ['yes', 'no', 'yes']
            # 先确保加上引号：['yes', 'no', 'yes']
            result_str = result_str.replace("yes", "'yes'").replace("no", "'no'")
            result_list = ast.literal_eval(result_str)

            for item in result_list:
                if item == "yes":
                    correct += 1
                    
        except openai.BadRequestError as e: # 非法输入 '$.input' is invalid. query返回结果为：请输入详细信息等
            print(e)
            total_num += 1
            continue

        except IndexError as e: # 得不到正确格式的query: set1=(fastest car)
            print(e)
            total_num += 1 # 防止卡死
            continue

        except openai.APITimeoutError as e: # 超时
            print(e)
            total_num += 1 # 防止卡死
            continue

        except ValueError as e: # maximum context length
            print(e)
            continue

        except Exception as e: # 其他错误
            print(e)
            total_num += 1 # 防止卡死
            continue

    # step 4: 粗略召回率判断：如果生成答案非空就算“尝试回答”
    for answer in answers:  
        if answer:
            matched += 1

    accuracy = correct / total
    recall = matched / total

    print(f"准确率 (Accuracy): {accuracy:.2%}")
    print(f"召回率 (Recall): {recall:.2%}")

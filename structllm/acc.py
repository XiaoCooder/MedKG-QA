import structllm as sllm
import json
from tqdm import tqdm
import openai
import ast
from sklearn.metrics import f1_score

def evaluate_answer_quality(args,data):
    llm = sllm.llm.gpt(args)

    total = len(data)
    total_num = 0
    correct = 0
    matched = 0
    answers = []
    p_answers = []
    p_questions = []
    for item in tqdm(data, desc="评估问答准确率"):
        question = item['Q']
        answer = item['A']
        p_answers.append(answer)
        p_questions.append(question)
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
    batch_size = 1
    all_results = []
    pred_labels = []
    true_labels = []

    for i in range(0, total, batch_size):
        batch_pred = p_answers[i:i + batch_size]
        batch_gold = answers[i:i + batch_size]
        batch_qs = p_questions[i:i + batch_size]
        batch_data = [{"pred": pred, "gold": gold} for pred, gold in zip(batch_pred, batch_gold)]

        check_prompt = sllm.query_prompt.query_prompt(args, batch_data)
        check_prompt.create_prompt(task="judge_acc", question=batch_qs)
        responses = llm.get_response(check_prompt.naive_prompt)

        for response in responses:
            try:
                result_str = response.choices[0].message.content.strip()
                result_str = result_str.replace("yes", "'yes'").replace("no", "'no'")
                result_list = ast.literal_eval(result_str)

                # DEBUG
                if len(result_list) != len(batch_pred):
                    print(f"[Warning] result_list size ({len(result_list)}) != batch_pred size ({len(batch_pred)})")

                all_results.append(result_list)

                for idx, item in enumerate(result_list):
                    if item == "yes":
                        correct += 1
                        pred_labels.append(1)
                        true_labels.append(1)
                    elif item == "no":
                        pred_labels.append(0)
                        true_labels.append(1)  # 真实标签默认都为1
                    else:
                        print(f"[Warning] Unexpected label: {item}")
                        # 跳过这个样本，避免污染

            except Exception as e:
                print(f"[Error] {e}")
                total_num += 1
                continue

        # step 4: 粗略召回率判断：如果生成答案非空就算“尝试回答”
        for answer in batch_gold:
            if answer:
                matched += 1
                
    """
    # 计算准确率和召回率
    accuracy = correct / total
    recall = matched / total

    # 防御性截断，避免长度不一致
    min_len = min(len(true_labels), len(pred_labels))
    f1 = f1_score(true_labels[:min_len], pred_labels[:min_len])
    """
    return correct, matched, pred_labels, true_labels
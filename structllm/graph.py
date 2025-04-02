import re
import chromadb
from typing import List, Tuple
import structllm as sllm
import openai
import os


def extract_keywords(args,questions_path):
    """
    利用大模型对输入数据进行关键词提取
    """
    questions = read_questions(questions_path)
    llm = sllm.llm.gpt(args)
    keywords_data = []
    total_num = 0
    flag = True
    max_retries = 3
    retry_count = 0
    while flag and retry_count < max_retries:
        retry_count += 1
        # 构造关键词提取的提示，task 设置为 "extract_keywords"
        query_prompt = sllm.query_prompt.query_prompt(args, questions)
        query_prompt.create_prompt(task="extract_keywords")

        # 调用大模型获取关键词及其类别
        responses_keywords = llm.get_response(query_prompt.naive_prompt)

        for response in responses_keywords:
            try:
                result = response.choices[0].message.content       # 关键词列表（格式：[['xxx', 'head', 'xxx', 'relation', 'xxx', 'tail'], [...], [...] ]）
                extracted_keywords = sllm.align.get_keywords(result)  

                for keyword_group in extracted_keywords:  # 遍历每个关键词组
                    keyword_dict = {"keyword_head": "", "keyword_relation": "", "keyword_tail": ""}  # 初始化为空字符串

                    for i in range(0, len(keyword_group), 2):  # 每两个元素一组
                        keyword, keyword_type = keyword_group[i], keyword_group[i + 1]

                        if keyword_type == "head":  
                            keyword_dict[f"keyword_{keyword_type}"] = keyword  # 存入关键词（即使是 ""）
                        if keyword_type == "relation":  
                            keyword_dict[f"keyword_{keyword_type}"] = keyword  # 存入关键词（即使是 ""）
                        if keyword_type == "tail":  
                            keyword_dict[f"keyword_{keyword_type}"] = keyword  # 存入关键词（即使是 ""）

                    keywords_data.append(keyword_dict)  # 存入最终的关键词字典

            except openai.BadRequestError as e:  # 处理无效输入
                print(e)
                total_num += 1
                continue


            except IndexError as e: 
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

            except Exception as e:
                print(f"提取关键词时出错: {e}")
                total_num += 1
                continue
            flag = False
    return keywords_data


def read_questions(questions_path):
    """
    读取 questions.txt 并返回问题列表

    参数：
        questions_path (str): 存储问题的文件路径

    返回：
        list: 包含所有问题的列表
    """
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"文件 {questions_path} 不存在！")

    questions = []
    try:
        with open(questions_path, 'r', encoding='utf-8') as file:
            for line in file:
                question = line.strip()
                if question:  # 跳过空行
                    questions.append(question)
    except Exception as e:
        print(f"读取文件时出错: {e}")

    return questions


def split_qa_pairs(file_path):
    """
    读取 qa_pairs.txt 并将问题和答案分开存入 questions.txt 和 answers.txt

    参数：
        file_path (str): qa_pairs.txt 的路径

    返回：
        tuple: (questions.txt 路径, answers.txt 路径)
    """
    qa_pairs_path = os.path.join(file_path, "qa_pairs.txt")

    if not os.path.exists(qa_pairs_path):
        raise FileNotFoundError(f"文件 {qa_pairs_path} 不存在")

    questions = []
    answers = []
    
    # 读取 qa_pairs.txt
    with open(qa_pairs_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 解析 Q 和 A
    for line in lines:
        line = line.strip()
        if line.startswith("Q: "):
            questions.append(line[3:].strip())  # 去掉 "Q: "
        elif line.startswith("A: "):
            answers.append(line[3:].strip())  # 去掉 "A: "

    # 生成新文件路径
    base_dir = os.path.dirname(qa_pairs_path)
    questions_path = os.path.join(base_dir, "questions.txt")
    answers_path = os.path.join(base_dir, "answers.txt")

    # 写入 questions.txt
    with open(questions_path, "w", encoding="utf-8") as fq:
        fq.write("\n".join(questions))

    # 写入 answers.txt
    with open(answers_path, "w", encoding="utf-8") as fa:
        fa.write("\n".join(answers))

    return questions_path, answers_path

class graph(): 
    def __init__(self, args, path):
        self.args = args
        self.path = path

    def load_triples(self, path):
        triples_path = os.path.join(path, "triples.txt")
        triples_head_set = set()
        triples_relation_set = set()
        triples_tail_set = set()
        triples_list = []
        chunk_pattern = "***"

        if not os.path.exists(triples_path):
            raise FileNotFoundError(f"triples.txt 不存在于路径: {path}")

        try:
            with open(triples_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"读取 {triples_path}，共 {len(lines)} 行")

                for line in lines:
                    line = line.strip()

                    # 跳过空行和分块标记行
                    if not line or line.startswith(chunk_pattern):
                        continue

                    # 你的数据格式是 [head, relation, tail]
                    # 所以需要去掉方括号，并按 ", " 逗号+空格 拆分
                    line = line.strip("[]")  # 去掉开头和结尾的 [ ]
                    parts = line.split(", ")  # 按 ", " 拆分

                    if len(parts) == 3:
                        head, relation, tail = parts
                        triples_list.append((head, relation, tail))
                        triples_head_set.add(head)
                        triples_relation_set.add(relation)
                        triples_tail_set.add(tail)
                    else:
                        print(f"无法解析行(拆分后长度不为 3): {line}")
        except Exception as e:
            print(f"读取文件时出错: {e}")
        
        triples_head_list = list(dict.fromkeys([t[0] for t in triples_list]))
        triples_relation_list = list(dict.fromkeys([t[1] for t in triples_list]))
        triples_tail_list = list(dict.fromkeys([t[2] for t in triples_list]))
        
        return triples_list, triples_head_list, triples_relation_list, triples_tail_list


def match_and_find_triples(args, path, keywords_data):
    """
    遍历 keywords_data 使用 SentenceBertRetriever 在 triples_list 中匹配合适的三元组。

    参数：
        path (str): 三元组数据的路径
        keywords_data (list of dict): 关键词列表，每个元素格式如下：
            [
                {"keyword_head": "xxx", "keyword_relation": "xxx", "keyword_tail": "xxx"},
                {"keyword_head": "yyy", "keyword_relation": "yyy", "keyword_tail": "yyy"},
                ...
            ]

    返回：
        list: 匹配到的所有三元组
    """

    # 加载三元组数据
    load_triples = graph(args,path)
    triples_list, triples_head_list, triples_relation_list, triples_tail_list = load_triples.load_triples(path)
    # 如果三元组列表为空，直接返回
    if not triples_list or not triples_head_list or not triples_relation_list or not triples_tail_list:
        return []

    # 初始化匹配器（只初始化非空列表）
    keywords_head_match = sllm.align.SentenceBertRetriever(triples_head_list) if triples_head_list else None
    keywords_relation_match = sllm.align.SentenceBertRetriever(triples_relation_list) if triples_relation_list else None
    keywords_tail_match = sllm.align.SentenceBertRetriever(triples_tail_list) if triples_tail_list else None

    matched_results = []  # 存放匹配到的三元组
    # 遍历 keywords_data 中的每个关键词字典
    for keyword_idx, keywords in enumerate(keywords_data):
        matched_head, matched_relation, matched_tail = None, None, None
        # 处理 head
        if keywords["keyword_head"].strip() and keywords_head_match:
            head_index_list = keywords_head_match.get_topk_candidates(1, keywords["keyword_head"])
            if head_index_list and isinstance(head_index_list, list):
                # 处理嵌套列表情况（如 [[0]]）
                if all(isinstance(item, list) for item in head_index_list):
                    head_index = head_index_list[0][0] if head_index_list[0] else None
                # 处理单层列表情况（如 [0]）
                elif all(isinstance(item, int) for item in head_index_list):
                    head_index = head_index_list[0]
                else:
                    head_index = None
                # 检查索引有效性
                if isinstance(head_index, int) and 0 <= head_index < len(triples_head_list):
                    matched_head = triples_head_list[head_index]

        # 处理 relation
        if keywords["keyword_relation"].strip() and keywords_relation_match:
            relation_index_list = keywords_relation_match.get_topk_candidates(1, keywords["keyword_relation"])
            if relation_index_list and isinstance(relation_index_list, list):
                # 处理嵌套列表情况（如 [[0]]）
                if all(isinstance(item, list) for item in relation_index_list):
                    relation_index = relation_index_list[0][0] if relation_index_list[0] else None
                # 处理单层列表情况（如 [0]）
                elif all(isinstance(item, int) for item in relation_index_list):
                    relation_index = relation_index_list[0]
                else:
                    relation_index = None
                # 检查索引有效性
                if isinstance(relation_index, int) and 0 <= relation_index < len(triples_relation_list):
                    matched_relation = triples_relation_list[relation_index]

        # 处理 tail
        if keywords["keyword_tail"].strip() and keywords_tail_match:
            tail_index_list = keywords_tail_match.get_topk_candidates(1, keywords["keyword_tail"])
            if tail_index_list and isinstance(tail_index_list, list):
                # 处理嵌套列表情况（如 [[0]]）
                if all(isinstance(item, list) for item in tail_index_list):
                    tail_index = tail_index_list[0][0] if tail_index_list[0] else None
                # 处理单层列表情况（如 [0]）
                elif all(isinstance(item, int) for item in tail_index_list):
                    tail_index = tail_index_list[0]
                else:
                    tail_index = None
                # 检查索引有效性
                if isinstance(tail_index, int) and 0 <= tail_index < len(triples_tail_list):
                    matched_tail = triples_tail_list[tail_index]
        
        current_matches = []
        for triple in triples_list:
            head, relation, tail = triple
            match = False
            
            # 情况1：仅head为None
            if matched_head is None and matched_relation is not None and matched_tail is not None:
                match = (relation == matched_relation) and (tail == matched_tail)
            
            # 情况2：仅relation为None
            elif matched_head is not None and matched_relation is None and matched_tail is not None:
                match = (head == matched_head) and (tail == matched_tail)
            
            # 情况3：仅tail为None
            elif matched_head is not None and matched_relation is not None and matched_tail is None:
                match = (head == matched_head) and (relation == matched_relation)
            
            # 情况4：全部不为None
            elif matched_head is not None and matched_relation is not None and matched_tail is not None:
                match = (head == matched_head) and (relation == matched_relation) and (tail == matched_tail)
            
            # 原始情况：全部为None（保留原有逻辑）
            # elif matched_head is None and matched_relation is None and matched_tail is None:
            #     match = True
            
            if match:
                current_matches.append(triple)
        matched_results.append({
            "keywords_index": keyword_idx,
            "keywords": keywords,
            "matched_triples": current_matches
        })
    return matched_results # 返回所有匹配到的三元组


def write_qa_and_matched_results_to_new_file(path, matched_results):
    """
    将 qa_pairs.txt 中的问题和答案以及 matched_results 一起写入到新创建的 txt 文件中。
    """
    # 创建新文件路径
    qa_pairs_path = os.path.join(path,"qa_pairs.txt")
    new_file_path = os.path.join(path, "qa_with_matched_results.txt")
    
    # 读取 qa_pairs.txt 文件
    qa_pairs = []
    try:
        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 将 QA 对按 "Q: " 和 "A: " 分开
            qa_pairs = content.strip().split("\n\n")  # 假设每一对问答中间有两个换行符
            
            # 对每一对 QA 进行处理，确保去掉每一行的 "Q: " 和 "A: "
            qa_pairs = [pair.replace("Q: ", "").replace("A: ", "") for pair in qa_pairs]
    except Exception as e:
        print(f"读取文件 {qa_pairs_file} 时出错: {e}")
        return
    
    # 创建并写入新文件
    try:
        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            # 遍历每个关键词索引
            for idx, matched_result in enumerate(matched_results):
                if idx >= len(qa_pairs):
                    break  # 如果 matched_results 的数量多于 QA 对数量，就跳出

                # 获取对应的 QA 对
                question_answer = qa_pairs[idx]
                question, answer = question_answer.split("\n", 1)

                # 写入问题和答案
                new_file.write(f"Q: {question}\n")
                new_file.write(f"A: {answer}\n")

                # 获取匹配的三元组
                matched_triples = matched_result.get('matched_triples', [])
                new_file.write(f"Matched Triples for this QA:\n")
                
                for triple in matched_triples:
                    new_file.write(f"  - {triple}\n")

                # 写入分隔符
                new_file.write("\n")

        print(f"已将结果成功写入文件: {new_file_path}")
        return new_file_path

    except Exception as e:
        print(f"写入文件时出错: {e}")
        return None


def triplesProcess(args, path):
    questions_path, answers_path = split_qa_pairs(path)
    keywords_data = extract_keywords(args,questions_path)
    matched_results = match_and_find_triples(args, path, keywords_data)
    write_qa_and_matched_results_to_new_file(path, matched_results)
    import pdb;pdb.set_trace()
    


    
    










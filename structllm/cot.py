import structllm as sllm
import openai

def cot(args, question, corpus, path):
    llm = sllm.llm.gpt(args)
    data = corpus.load_triples(path)
    #获取三元组列表
    triples_list = data[0]
    triples_text = "\n".join([f"{h},{r},{t}" for h, r, t in triples_list])
    #把三元组的内容和question输入模型进行问答   
    answers = []
    used_triples = []
    total_num = 0
    flag = True
    max_retries = 3
    retry_count = 0
    while flag and retry_count < max_retries:
        retry_count += 1
        query_prompt = sllm.query_prompt.query_prompt(args,triples_text)
        # 构造关键词提取的提示，task 设置为 "get_answer"
        query_prompt.create_prompt(task = "get_answer", question = question) 
        responses = llm.get_response(query_prompt.naive_prompt)
        for response in responses:
            try:
                result = response.choices[0].message.content
                answer_response, used_triples_response = sllm.align.get_answer_and_triples(result)
                if answer_response and answer_response not in answers:
                    answers.append(answer_response)

            # 添加三元组（一个答案可能涉及多个三元组）
                if used_triples_response:
                    if isinstance(used_triples_response[0], list):  # 如果是多个三元组
                        for triple in used_triples_response:
                            if triple not in used_triples:
                                used_triples.append(triple)
                    else:  # 只有一个三元组
                        if used_triples_response not in used_triples:
                            used_triples.append(used_triples_response)             
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
            flag = False
        #import pdb;pdb.set_trace()        
    return answers, used_triples
    #输出答案和三元组列表

    
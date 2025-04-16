import structllm as sllm
import openai
import asyncio

async def cot(args, question, corpus):
    llm = sllm.llm.gpt(args)
    #获取该问题匹配到的三元组列表
    matched_results = sllm.graph.triplesProcess(args, corpus, question)
    matched_triples = []
    for item in matched_results:
        for triple in item["matched_triples"]:
            matched_triples.append(triple)
    print(matched_triples)
    triples_text = ",".join([f"[{h},{r},{t}]" for h, r, t in matched_triples])
    #把三元组的内容和question输入模型进行问答   
    answers = []
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
                #answer_response = sllm.align.get_answer_and_triples(result)
                if result and result not in answers:
                    answers.append(result)
             
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
    return answers, matched_triples


    
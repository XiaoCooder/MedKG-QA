import structllm as sllm
import openai

def cot(args, question, corpus):
    llm = sllm.llm.gpt(args)
    chunk_id = rerank[0]
    
    #获取三元组列表
    #把三元组的内容和question输入模型进行问答
    #输出答案和三元组列表

    results = sllm.retrieve.get_summary_collection_and_query_chunk(args.encoder_model, chunk_id)
    #import pdb; pdb.set_trace()
    chunk_data = [results['metadatas'][0].get('chunk_data', '')]
    query_prompt = sllm.query_prompt.query_prompt(args,chunk_data)
    query_prompt.create_prompt(task = "get_answer", question = question)
    responses = llm.get_response(query_prompt.naive_prompt)
    for response in responses:
                try:
                    result = response.choices[0].message.content
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
                
    return result
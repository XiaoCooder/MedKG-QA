import re
import structllm as sllm
import json
import openai

def rerank(args, question):
    llm = sllm.llm.gpt(args)

    results = sllm.retrieve.get_summary_collection_and_query(args.encoder_model, query_texts = question , recall_num = 5)
    summarys= [candidate_question for candidate_question in results['documents'][0]]
    chunk_ids = [candidate_content.get('chunk_id', '') for candidate_content in results['metadatas'][0]]
    flag = True
    while (flag) :
        ########1.Summary rerank#######
        query_prompt = sllm.query_prompt.query_prompt(args, summarys)
        query_prompt.create_prompt(task = "summary_rerank", question = question)
        responses = llm.get_response(query_prompt.naive_prompt)
        for response in responses:
            try:
                #解析response_sum
                result = response.choices[0].message.content
                summary_rerank = sllm.align.get_chunk_id(result, chunk_ids)           
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
    
    results_prompt = sllm.retrieve.get_context_collection_and_query(args.encoder_model, query_texts = question , recall_num = 5)
    flag = True
    while (flag) :
        ########1.Summary detection#######
        query_prompt = sllm.query_prompt.query_prompt(args, summarys)
        query_prompt.create_prompt(task = "context_rerank", question = question)
        responses = llm.get_response(query_prompt.naive_prompt)
        for response in responses:
            try:
                #解析response_sum
                result = response.choices[0].message.content
                context_rerank = sllm.align.get_chunk_id(result, chunk_ids)           
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

    qas = sllm.retrieve.get_qas_collection_and_query(args.encoder_model, query_texts = question , recall_num = 5)
    flag = True
    while (flag) :
        ########1.Summary detection#######
        query_prompt = sllm.query_prompt.query_prompt(args, summarys)
        query_prompt.create_prompt(task = "qas_rerank", question = question)
        responses = llm.get_response(query_prompt.naive_prompt)
        for response in responses:
            try:
                #解析response_sum
                result = response.choices[0].message.content
                
                qas_rerank = sllm.align.get_chunk_id(result, chunk_ids)           
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

    ###rerank fusion###
    rerank_result = None

    return rerank_result, context_rerank, summary_rerank, qas_rerank 
    

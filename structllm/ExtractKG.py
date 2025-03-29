import re
import structllm as sllm
import json
import openai

def ExtractKG1(args, data):
    """
    利用大模型对输入数据进行三元组提取，并将提取的三元组保存到数据库中。
    """
    #args 系统设置
    #data 保存在列表中的文本内容

    # 初始化语言模型（如 GPT 模型）
    llm = sllm.llm.gpt(args)
    # 用于存储提取出的三元组数据，格式为列表，每个元素为 [a, b, c]
    triple_data = []

    total_num = 0  # 用于错误累计，防止卡死

    # 利用大模型提取三元组
    flag = True
    max_retries = 3
    retry_count = 0
    while flag and retry_count < max_retries:
        retry_count += 1
        # 构造三元组提取的提示，task 设置为 "extract_triple"
        query_prompt = sllm.query_prompt.query_prompt(args, data)
        query_prompt.create_prompt(task="extract_triple")
        responses_triple = llm.get_response(query_prompt.naive_prompt)
        for response in responses_triple:
            try:
                result = response.choices[0].message.content
                # 假设此处返回内容为形如 "[a, b, c]" 的字符串，可以通过自定义函数解析提取成列表形式
                # 例如，使用 sllm.align.get_triples(result) 解析返回三元组列表
                triple_data = sllm.align.get_triples(result)
            except openai.BadRequestError as e: # 非法输入 '$.input' is invalid. query返回结果为：请输入详细信息等
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
                print(f"提取三元组时出错: {e}")
                total_num += 1
                continue
            flag = False

    # 将提取的三元组数据写入数据库
    # 这里调用自定义的数据库写入函数，需保证 args.encoder_model 和 chunk_id 参数有效
    sllm.retrieve.get_triples_collection_and_write(args.encoder_model, triple_data=triple_data)


    # 也将上下文数据存入数据库，便于后续检索
    #sllm.retrieve.get_context_collection_and_write(args.encoder_model, context_data=data)

    return triple_data

def ExtractKG(args, data):
    """
    利用大模型对输入数据进行三元组提取，并将提取的三元组以字典形式存储，
    三元组中的 head、relation、tail 分别写入 chromadb 中。
    """
    # args 系统设置
    # data 保存在列表中的文本内容

    # 初始化语言模型（如 GPT 模型）
    llm = sllm.llm.gpt(args)
    # 用于存储提取出的三元组数据，格式为列表，每个元素为字典：
    # {"head": value, "relation": value, "tail": value}
    triple_data = []

    total_num = 0  # 用于错误累计，防止卡死

    # 利用大模型提取三元组
    flag = True
    max_retries = 3
    retry_count = 0
    while flag and retry_count < max_retries:
        retry_count += 1
        # 构造三元组提取的提示，task 设置为 "extract_triple"
        query_prompt = sllm.query_prompt.query_prompt(args, data)
        query_prompt.create_prompt(task="extract_triple")
        responses_triple = llm.get_response(query_prompt.naive_prompt)
        #print(responses_triple)
        for response in responses_triple:
            try:
                result = response.choices[0].message.content
                # 假设返回的内容为形如 "[[head1, relation1, tail1], [head2, relation2, tail2], ...]" 的字符串，
                # 通过自定义函数解析成列表形式
                # 修改后的三元组处理逻辑
                raw_triples = sllm.align.get_triples(result)
                for triple in raw_triples:
                    if len(triple) == 3:
                        triple_dict = {
                            "head": str(triple[0]).strip() if triple[0] else None,
                            "relation": str(triple[1]).strip() if triple[1] else None,
                            "tail": str(triple[2]).strip() if triple[2] else None
                        }
                        # 严格验证所有字段
                        if all(triple_dict.values()):  # 确保没有None或空字符串
                            triple_data.append(triple_dict)
                        else:
                            print(f"跳过无效三元组（存在空值）: {triple}")
                    else:
                        print(f"警告:忽略长度不为3的三元组: {triple}")
                        
            except openai.BadRequestError as e:  # 非法输入
                print(e)
                total_num += 1
                continue
            except IndexError as e:
                print(e)
                total_num += 1  # 防止卡死
                continue
            except openai.APITimeoutError as e:  # 超时
                print(e)
                total_num += 1  # 防止卡死
                continue
            except ValueError as e:  # maximum context length
                print(e)
                continue
            except Exception as e:
                print(f"提取三元组时出错: {e}")
                total_num += 1
                continue
      
        for triple_dict in triple_data:
            # 将字典转换回列表
            restored_triple = [triple_dict["head"], triple_dict["relation"], triple_dict["tail"]]
    
            # 以一行字符串形式表示
            triple_str = f"[{restored_triple[0]}, {restored_triple[1]}, {restored_triple[2]}]"
            # 添加到 data 列表
            data.append(triple_str)
    print(len(data))
    
    sllm.retrieve.get_triples_head_collection_and_write(args.encoder_model, data=triple_data)
    sllm.retrieve.get_triples_relation_collection_and_write(args.encoder_model, data=triple_data)
    sllm.retrieve.get_triples_tail_collection_and_write(args.encoder_model, data=triple_data)
    # 如有需要，也将上下文数据存入数据库，便于后续检索
    # sllm.retrieve.get_context_collection_and_write(args.encoder_model, context_data=data)

    return triple_data

async def ExtractKGQA(args, data, encoder):
    """
    1. 利用大模型对输入数据进行三元组提取，并以字典形式存储；
    2. 利用大模型对输入数据进行 QA 问答对的提取；
    3. 将三元组的 head、relation、tail 以及 QA 问答对分别存入 chromadb。
    """
    # 初始化语言模型（如 GPT 模型）
    llm = sllm.llm.gpt(args)

    # 存储三元组数据，每个元素为 {"head": value, "relation": value, "tail": value}
    triple_data = []

    # 存储 QA 问答对，每个元素为 {"question": value, "answer": value}
    qa_data = []

    total_num = 0  # 统计错误次数，防止死循环
    max_retries = 3  # 最大重试次数
    retry_count = 0

    # === 1. 提取三元组 ===
    flag = True
    while flag and retry_count < max_retries:
        retry_count += 1
        query_prompt = sllm.query_prompt.query_prompt(args, data)
        query_prompt.create_prompt(task="extract_triple")
        responses_triple = llm.get_response(query_prompt.naive_prompt)

        for response in responses_triple:
            try:
                result = response.choices[0].message.content
                raw_triples = sllm.align.get_triples(result)
                for triple in raw_triples:
                    if len(triple) == 3:
                        triple_dict = {
                            "head": str(triple[0]).strip() if triple[0] else None,
                            "relation": str(triple[1]).strip() if triple[1] else None,
                            "tail": str(triple[2]).strip() if triple[2] else None
                        }
                        if all(triple_dict.values()):  # 确保三元组所有字段都有效
                            triple_data.append(triple_dict)
                        else:
                            print(f"跳过无效三元组（存在空值）: {triple}")
                    else:
                        print(f"警告: 忽略长度不为3的三元组: {triple}")

            except Exception as e:
                print(f"提取三元组时出错: {e}")
                total_num += 1
                continue
            flag = False

    
    for triple_dict in triple_data:
        # 将字典转换回列表
        restored_triple = [triple_dict["head"], triple_dict["relation"], triple_dict["tail"]]
        # 以一行字符串形式表示
        triple_str = f"[{restored_triple[0]}, {restored_triple[1]}, {restored_triple[2]}]"
        # 添加到 data 列表
        data.append(triple_str)


    # === 2. 提取 QA 问答对 ===
    retry_count = 0  # 重置重试计数
    flag = True
    while flag and retry_count < max_retries:
        retry_count += 1
        query_prompt = sllm.query_prompt.query_prompt(args, data)
        query_prompt.create_prompt(task="extract_qa")
        responses_qa = llm.get_response(query_prompt.naive_prompt)
        for response in responses_qa:
            try:
                result = response.choices[0].message.content
                raw_qa_pairs = sllm.align.get_qa_pairs(result)  
                for qa_pair in raw_qa_pairs:
                    if len(qa_pair) == 2:
                        qa_dict = {
                            "question": str(qa_pair[0]).strip() if qa_pair[0] else None,
                            "answer": str(qa_pair[1]).strip() if qa_pair[1] else None
                        }
                        if all(qa_dict.values()):  # 确保 Q & A 都有效
                            qa_data.append(qa_dict) 
                        else:
                            print(f"跳过无效问答对（存在空值）: {qa_pair}")
                    else:
                        print(f"警告: 忽略长度不为2的问答对: {qa_pair}")
            except Exception as e:
                print(f"提取 QA 问答对时出错: {e}")
                total_num += 1
                continue
            flag = False

    # === 3. 写入 chromadb ===

        # 分别存储 head、relation、tail
    await sllm.retrieve.get_triples_head_collection_and_write(data=triple_data, encoder = encoder)
    await sllm.retrieve.get_triples_relation_collection_and_write(data=triple_data, encoder = encoder)
    await sllm.retrieve.get_triples_tail_collection_and_write(data=triple_data, encoder = encoder)

        # 存储 QA 问答对
    await sllm.retrieve.get_qa_collection_and_write(data=qa_data, encoder = encoder)

    return triple_data, qa_data


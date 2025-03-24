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
        for response in responses_triple:
            try:
                result = response.choices[0].message.content
                # 假设返回的内容为形如 "[[head1, relation1, tail1], [head2, relation2, tail2], ...]" 的字符串，
                # 通过自定义函数解析成列表形式
                raw_triples = sllm.align.get_triples(result)
                # 将每个三元组转换为字典形式存储
                for triple in raw_triples:
                    if len(triple) == 3:
                        triple_dict = {
                            "head": triple[0],
                            "relation": triple[1],
                            "tail": triple[2]
                        }
                        triple_data.append(triple_dict)
                    else:
                        print("解析得到的三元组长度不为 3:", triple)
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
            flag = False

    # 将提取的三元组数据写入数据库（如果还需要写入其他数据库可以保留此步骤）
    #sllm.retrieve.get_triples_collection_and_write(args.encoder_model, triple_data=triple_data)

    # 分别将三元组中的 head、relation、tail 存入 chromadb 中
    for triple in triple_data:
        try:
            if triple is None:
                print("警告: triple 为 None, 跳过处理")
                continue
            
            # 确保所有字段存在且为非空字符串
            head = triple.get("head")
            if not head:  # 检查空字符串或None
                print(f"跳过无效的三元组，缺少有效的 head: {triple}")
                continue
            
            relation = triple.get("relation")
            if not relation:
                print(f"跳过无效的三元组，缺少有效的 relation: {triple}")
                continue
            
            tail = triple.get("tail")
            if not tail:
                print(f"跳过无效的三元组，缺少有效的 tail: {triple}")
                continue
            
            # 处理有效三元组
            sllm.retrieve.get_triples_head_collection_and_write(args.encoder_model, data=head)
            sllm.retrieve.get_triples_relation_collection_and_write(args.encoder_model, value=relation)
            sllm.retrieve.get_triples_tail_collection_and_write(args.encoder_model, value=tail)
        
        except Exception as e:
            print(f"写入 chromadb 时出错: {e}")
            continue
        
    # 如有需要，也将上下文数据存入数据库，便于后续检索
    # sllm.retrieve.get_context_collection_and_write(args.encoder_model, context_data=data)

    return triple_data


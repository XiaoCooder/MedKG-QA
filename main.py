# main.py
import json 
from tqdm import tqdm
import argparse
import os
import structllm as sllm
import re
import asyncio
from aiomultiprocess import Pool
from sentence_transformers import SentenceTransformer
import torch
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Images
import settings  # 导入共享设置模块
import threading  # 导入线程模块
import app  # 导入 Flask 应用
import time  # 导入时间模块
from sklearn.metrics import f1_score, recall_score, accuracy_score

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 全局变量，用于线程间通信
flask_ready = False
settings_ready = threading.Event()

# 启动 Flask 服务的线程函数
def run_flask_app(port=5000):
    global flask_ready
    print(f"启动 Flask 服务，请访问 http://localhost:{port} 设置 URL 和 API")
    flask_ready = True  # 在启动 Flask 之前设置为 True
    app.start_flask(port=port)

# 应用用户设置到 args
def apply_user_settings(args):
    """应用用户在 Flask 中设置的 URL 和 API"""
    url = settings.get_url()
    api = settings.get_api()
    
    if url:
        args.openai_url = url
        print(f"使用设置的 URL: {url}")
    
    if api:
        args.key = api
        print(f"使用设置的 API: {api}")
    
    return args

# 监视设置变化
def wait_for_settings():
    print("等待通过前端设置 URL 和 API...")
    
    while True:
        url = settings.get_url()
        api = settings.get_api()
        
        if url and api:
            print("检测到设置已完成！")
            settings_ready.set()  # 设置事件，通知主线程
            break
        
        time.sleep(1)  # 每秒检查一次

# 检查路径是否存在，如果不存在则创建目录
def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

# 批量处理数据以提取知识图谱
async def KGProcess(args, Data, idx, api_key , encoder):
    
                args.key = api_key
                
                if idx == -1:
                        output_path= args.output_path
                        describe = "process"
                else:
                        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
                        check_path(args.output_path)
                        output_path = args.output_path + "/p-" + idx
                        describe = "p-" + idx
                        check_path(output_path)
                        
                print("Start PID %d and save to %s" % (os.getpid(), output_path))
                
                if not args.debug:
                    try:
                        for i in range(len(Data) // args.batch_size + (1 if len(Data) % args.batch_size != 0 else 0)):
                          start_index = i * args.batch_size
                          end_index = min(start_index + args.batch_size, len(Data))  # 确保不超过总长度
                          # 提取一个批次
                          subData = Data[start_index:end_index]
                          triple_data, qa_data= await sllm.ExtractKG.ExtractKGQA(args,subData,encoder)
                    except Exception as e:    
                        if args.store_error:
                            pass

                else:
                    triple_data_path = os.path.join(output_path, 'triples.txt')
                    qa_data_path = os.path.join(output_path, 'qa_pairs.txt')
                    check_path(triple_data_path)
                    check_path(qa_data_path)
                    num_batches = len(Data) // args.batch_size + (1 if len(Data) % args.batch_size != 0 else 0)
                    
                    for i in tqdm(range(num_batches), desc = describe):
                        start_index = i * args.batch_size
                        end_index = min(start_index + args.batch_size, len(Data))  # 确保不超过总长度
                          # 提取一个批次
                        subData = Data[start_index:end_index]
                        
                        triple_data, qa_data = await sllm.ExtractKG.ExtractKGQA(args, subData, encoder)
                        #save chunk
                        with open(triple_data_path, 'a', encoding='utf-8') as fout:
                                for triple in triple_data:
                                    fout.write(f"[{triple['head']}, {triple['relation']}, {triple['tail']}]\n") 
                        with open(qa_data_path, 'a', encoding='utf-8') as fout:
                                for qa in qa_data:
                                    fout.write(f"Q: {qa['question']}\nA: {qa['answer']}\n\n")  # 问答对格式化                          

# 批量处理数据以提取知识图谱
async def EvalProcess(args, data, idx, api_key):
    args.key = api_key              
    if not args.debug:
        try:
            for i in range(len(data) // args.batch_size + (1 if len(data) % args.batch_size != 0 else 0)):
                start_index = i * args.batch_size
                end_index = min(start_index + args.batch_size, len(data))  # 确保不超过总长度
                # 提取一个批次
                subData = data[start_index:end_index]
                correct, matched, pred_labels, true_labels = await sllm.acc.evaluate_answer_quality(args,subData)
        except Exception as e:    
            if args.store_error:
                pass

    else:
        correct, matched, pred_labels, true_labels = await sllm.acc.evaluate_answer_quality(args, data)
    
    return correct, matched, pred_labels, true_labels
                    
                                    
# 读取并解析文本数据为完整句子
def TxtRead(args):
    print('load txt data...')
    with open(args.data_path, "r", encoding="utf8") as fin:
        raw_lines = fin.readlines()

    # 先去除空行和两端空白
    lines = [line.strip() for line in raw_lines]

    sentences = []  # 存储合并后的完整句子
    current_sentence = ""  # 用来暂存当前正在组合的句子

    # 定义句子结束的标点符号，可根据需要进行扩展
    sentence_endings = ("。", "!", "?")

    for line in lines:
        if current_sentence:
            # 句子拼接时在行间添加空格
            current_sentence += " " + line
        else:
            current_sentence = line

        # 如果当前行以句子结束标点结尾，则认为是完整句子
        if line.endswith(sentence_endings):
            sentences.append(current_sentence.strip())
            current_sentence = ""

    # 如果文件最后一部分没有以结束标点结束，也将其作为一句
    if current_sentence:
        sentences.append(current_sentence.strip())

    print(f"文本共 {len(sentences)} 个完整句子")
    return sentences

def TxtRead1(args):
    print('load txt data...')
    sentences = []
    with open(args.data_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 去除前缀编号（如 1. 2.）
            if line[0].isdigit():
                dot_index = line.find('.')
                if 0 < dot_index < 4:
                    line = line[dot_index + 1:].strip()
            sentences.append(line)
    return sentences

# 解析应用程序的命令行参数
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    # setting for openai
    parser.add_argument('--openai_url', default="", type=str, help='The url of openai')
    parser.add_argument('--key', default="api_key.txt", type=str, help='The key of openai or path of keys')
    parser.add_argument('--embedding_key', default="", type=str, help='The key of openai or path of keys')
    parser.add_argument('--dynamic_open', default=True, type=bool, help='The key of openai or path of keys')

    # input data path
    parser.add_argument('--folder_path', default="dataset/WikiSQL_TB_csv/test", type=str, help='The CSV data pth.')
    parser.add_argument('--data_path', default="dataset/WikiSQL_CG", type=str, help='The CG data pth.')
    parser.add_argument('--character_path', default="input/character.txt", type=str, help='')
    
    #prompt path
    parser.add_argument('--clean_prompt_path', default="structllm/prompt_/clean_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--extract_q_prompt_path', default="structllm/prompt_/extract_q_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--extract_a_prompt_path', default="structllm/prompt_/extract_a_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--summary_prompt_path', default="structllm/prompt_/summary_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--reranker_prompt', default="structllm/prompt_/reranker_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--qa_prompt', default="structllm/prompt_/qa_prompt.json", type=str, help='The prompt pth.')
    parser.add_argument('--extractKG_prompt', default="structllm/prompt_/extractKG.json", type=str, help='The prompt pth.')
    parser.add_argument('--extractQA', default="structllm/prompt_/extractKGQA.json", type=str, help='The prompt pth.')
    parser.add_argument('--extract_keywords', default="structllm/prompt_/extract_keywords.json", type=str, help='The prompt pth.')
    parser.add_argument('--get_answer', default="structllm/prompt_/get_answer_and_triple.json", type=str, help='The prompt pth.')
    parser.add_argument('--get_answer1', default="structllm/prompt_/get_answer.json", type=str, help='The prompt pth.')
    parser.add_argument('--acc_prompt', default="structllm/prompt_/judge_acc.json", type=str, help='The prompt pth.')

    # setting model
    parser.add_argument('--model', default="gpt-3.5-turbo", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    parser.add_argument('--encoder_model', default="SentenceBERT", type=str, help='The openai model. "gpt-3.5-turbo-0125" and "gpt-4-1106-preview" are supported')
    parser.add_argument('--retriever_align', default="SentenceBERT", type=str, help='')
    parser.add_argument('--batch_size', default="10", type=int, help='The prompt pth.')
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')
    
    # output
    parser.add_argument('--output_path', default="output/output_result.txt", type=str)
    parser.add_argument('--qa_output_path', default="output/qa_history.txt", type=str)

    #others
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--flask_port', default=5000, type=int, help='Flask 服务端口')
    parser.add_argument('--test', default=True, type=bool)

    
    
    args = parser.parse_args()
    return args

# 合并多线程产生的
def merge_output_files(base_dir, num_process):
    merged_dir = os.path.join(base_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    merged_qa_path = os.path.join(merged_dir, "qa_pairs.txt")
    merged_triples_path = os.path.join(merged_dir, "triples.txt")

    with open(merged_qa_path, 'w', encoding='utf-8') as qa_out, \
         open(merged_triples_path, 'w', encoding='utf-8') as triple_out:

        for idx in range(num_process):
            sub_dir = os.path.join(base_dir, f"p-{idx:02d}")
            qa_file = os.path.join(sub_dir, "qa_pairs.txt")
            triple_file = os.path.join(sub_dir, "triples.txt")

            # 合并 qa_pairs
            if os.path.exists(qa_file):
                with open(qa_file, 'r', encoding='utf-8') as f:
                    qa_out.writelines(f.readlines())

            # 合并 triples
            if os.path.exists(triple_file):
                with open(triple_file, 'r', encoding='utf-8') as f:
                    triple_out.writelines(f.readlines())

    print(f"合并完成，输出路径：{merged_qa_path}, {merged_triples_path}")


# 应用程序的主入口
async def main():
    args = parse_args()
    
    # 在单独的线程中启动 Flask 应用
    flask_thread = threading.Thread(target=run_flask_app, args=(args.flask_port,))
    flask_thread.daemon = True  # 设置为守护线程
    flask_thread.start()
    
    # 等待 Flask 服务启动准备就绪
    time.sleep(1)  # 给 Flask 线程一点时间来设置 flask_ready
    
    print(f"\n======================================================")
    print(f"Flask 服务已启动，请访问 http://localhost:{args.flask_port} 设置 URL 和 API")
    print(f"======================================================\n")
    
    # 在另一个线程中监视设置变化
    settings_thread = threading.Thread(target=wait_for_settings)
    settings_thread.daemon = True
    settings_thread.start()
    
    # 等待设置完成
    if not settings.get_url() or not settings.get_api():
        print("程序将在设置完成后继续...")
        settings_ready.wait()  # 等待设置完成
    
    # 应用用户设置
    args = apply_user_settings(args)
    
    print(args.openai_url)
    print(args.key)
    print("\n设置已应用，继续执行主程序...\n")
    
    # 初始化编码器的状态
    settings.set_encoder_ready(False)
    print("开始加载编码器模型，这可能需要一些时间...")
    
    # 耗时操作：加载编码器
    # 耗时操作：加载编码器
    try:
        encoder = sllm.retrieve.Encoder(args.encoder_model)
        # 加载完成后，仅设置状态为就绪
        settings.set_encoder_ready(True)
        print("编码器模型加载完成！")
    except Exception as e:
        print(f"加载编码器时出错: {str(e)}")
        # 即使出错，也需要更新状态以避免前端一直等待
        settings.set_encoder_ready(True)
        raise e

    print("start data choice")
    # 清除可能存在的之前的选择
    settings.clear_data_choice()

    # 默认有数据，引导用户到选择页面
    print(f"\n======================================================")
    print(f"请访问 http://localhost:{args.flask_port}/data_choice 选择如何继续")
    print(f"======================================================\n")

    # 等待用户在网页上做出选择
    while settings.get_data_choice() is None:
        print("等待用户选择是否直接加载已有数据...")
        time.sleep(2)  # 每2秒检查一次

    user_input = settings.get_data_choice()
    print(f"用户选择了: {user_input}")

    #load api-key
    if not args.key.startswith("sk-"):
        with open(args.key, "r",encoding='utf-8') as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) >= args.num_process, (len(all_keys), args.num_process)
    else:
        all_keys = []
        all_keys.append(args.key)
    
    if user_input == "no":
        #create DB （如果第一次安装可能没有数据）
        await sllm.retrieve.get_collection(name="path", encoder=encoder)
        await sllm.retrieve.get_collection(name="triples", encoder=encoder)
        await sllm.retrieve.get_collection(name="triple_head", encoder=encoder)
        await sllm.retrieve.get_collection(name="triple_relation", encoder=encoder)
        await sllm.retrieve.get_collection(name="triple_tail", encoder=encoder)
        await sllm.retrieve.get_collection(name="qa_pairs", encoder=encoder)
        #reset DB （避免遗留数据）
        await sllm.retrieve.rebuild_collection(name="path", encoder=encoder)
        await sllm.retrieve.rebuild_collection(name="triples", encoder=encoder)
        await sllm.retrieve.rebuild_collection(name="triple_head", encoder=encoder)
        await sllm.retrieve.rebuild_collection(name="triple_relation", encoder=encoder)
        await sllm.retrieve.rebuild_collection(name="triple_tail", encoder=encoder)
        await sllm.retrieve.rebuild_collection(name="qa_pairs", encoder=encoder)
        

        # 设置数据处理状态为处理中
        settings.set_process_data_ready(False)

        #load interview data
        data = TxtRead(args)
        # data_path_ceshi = "/home/wcy/code/KG-MedQA-v1.0/data_ceshi"
        # with open(data_path_ceshi, 'w', encoding='utf-8') as f:
        #      for d in data:
        #           f.write(f"{d}\n")
        # import pdb;pdb.set_trace()
        #process
        print(args.num_process)
        if args.num_process == 1:
            print("args.num_process == 1")
            await KGProcess(args, data, -1, all_keys[0] if 'all_keys' in locals() else args.key, encoder)
        else:
            num_each_split = int(len(data) / args.num_process)
            split_data = []
            for idx in range(args.num_process):
                    start = idx * num_each_split
                    if idx == args.num_process - 1:
                        end = max((idx + 1) * num_each_split, len(data))
                        split_data.append(data[start:end])
                    else:
                        end = (idx + 1) * num_each_split
                        split_data.append(data[start:end])
            async with Pool() as pool:
                    tasks = [pool.apply(KGProcess, args=(args, split_data[idx], idx, all_keys[idx], encoder)) for idx in range(args.num_process)]
                    await asyncio.gather(*tasks)
            #merge txt
            #merge_chunks(args)
        
        # 处理完成后，更新状态
        settings.set_process_data_ready(True)

        #Q&A system
        await sllm.retrieve.get_path_collection_and_write(path=args.output_path, encoder=encoder)
        args.qa_output_path = os.path.join(args.output_path, 'qa_history.txt')
        qa_bot = sllm.user_qa.user_qa(args)
        settings.set_qa_bot(qa_bot)
        print(f"\n问答系统已就绪，您也可以在网页界面中提问")
        print(f"访问 http://localhost:{args.flask_port}/qa 开始问答\n")
        qa_bot.start()  
        
    elif user_input == "yes":

        settings.set_load_data_ready(False)

        #Q&A system
        response = await sllm.retrieve.get_output_path(encoder=encoder)  # 先 await 获取返回值
        path = [candidate_content.get('path') for candidate_content in response['metadatas'][0]][0]
        current_path = os.path.join(path,'merged')
        print(path)
        print(current_path)
        qa_pairs_path = os.path.join(current_path,'qa_pairs.txt')
        print(qa_pairs_path)
        qs_path = sllm.graph.split_qa_pairs(qa_pairs_path)
        print(qs_path)
        args.qa_output_path = os.path.join(current_path, 'qa_history.json')
   
        #读取数据库内容
        corpus_ = sllm.graph.graph(args,current_path)
        corpus = corpus_.load_triples()
        qa_bot = sllm.user_qa.user_qa(args, corpus, current_path, qs_path)
        settings.set_qa_bot(qa_bot)
        # 加载完成后，更新状态
        settings.set_load_data_ready(True)
        print(f"\n问答系统已就绪，您也可以在网页界面中提问")
        print(f"访问 http://localhost:{args.flask_port}/qa 开始问答\n")
        qa_bot.start() 

if __name__=="__main__":
    asyncio.run(main())
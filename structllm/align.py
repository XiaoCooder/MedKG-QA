import re
import structllm as sllm
import ast
from sentence_transformers import SentenceTransformer,util
import torch
import re
import json
           
class SentenceBertRetriever:
    def __init__(self, corpus) -> None:

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        # self.device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
        self.retrieve_model = SentenceTransformer(
            'MiniCPM-Embedding-Light',
            device=self.device ,
            trust_remote_code=True,
        )
        self.corpus = corpus

    def get_embedding(self, text):
        result = self.retrieve_model.encode(text)
        flat_array = result.tolist()
        return flat_array

    def get_topk_candidates(self, topk, query):
        # Corpus of documents and their embeddings
        api_corpus_embeddings = self.get_embedding(self.corpus)
        api_corpus_embeddings_list = []
        for i in range(len(api_corpus_embeddings)):
            api_corpus_embeddings_list.append(api_corpus_embeddings[i])
        # Queries and their embeddings
        queries_embeddings = self.get_embedding(query)
        queries_embeddings_list = []
        for i in range(len(queries_embeddings)):
            queries_embeddings_list.append(queries_embeddings[i])
        # Find the top-k corpus documents matching each query
        cos_scores = util.cos_sim(queries_embeddings_list, api_corpus_embeddings_list)
        # hits = util.semantic_search(queries_embeddings, corpus_embeddings, top_k=topk)
        all_query_candidate_api_index = []
        for i in range(len(cos_scores)):
            hits = torch.argsort(cos_scores[i], descending=True)[:topk]
            all_query_candidate_api_index.append(hits.tolist())
        return all_query_candidate_api_index


    def count_accuracy(self, label, candidate):
        assert len(label) == len(candidate)

        topk_count = 0
        # hit = [0]*30
        # count = [0]*30
        for i in range(len(label)):
            # count[label[i]] += 1
            if label[i] in candidate[i]:
                topk_count += 1
                # hit[label[i]] += 1
        accuracy = topk_count / len(label)
        return accuracy

def GetRetriever(args, corpus):
    if args.retriever_align == "SentenceBERT":
        retriever = SentenceBertRetriever(corpus)
    else:
        pass
    return retriever

def get_parameters(text):
    names = []
    questions = []

    # 使用正则表达式分割数据项
    pattern = r"\{(.*?):(.*?)\}"

    matches = re.findall(pattern, text)

    # 提取姓名和问题
    for match in matches:
       names.append(match[0].strip())  # 姓名
       questions.append(match[1].strip())  # 问题
    return names, questions

def get_triples(text):
    """
    使用正则表达式从大模型生成的回答中提取三元组。
    
    参数：
        text (str): 大模型的回答文本，其中包含形如 [a, b, c] 的三元组
    
    返回：
        list: 三元组列表，每个三元组为一个包含三个字符串的列表
    """
    # 定义正则表达式：
    # \[ 和 \] 匹配中括号
    # (.*?) 使用非贪婪匹配提取每个元素，逗号为分隔符
    pattern = r'\[\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*\]'
    # 使用 re.findall 提取所有匹配的三元组，返回的结果是一个元组列表
    matches = re.findall(pattern, text)
    # 将元组转换为列表形式，并去除每个元素两边的空白字符
    triples = [[item.strip() for item in match] for match in matches]
    return triples

def get_qa_pairs(text):
    """
    使用正则表达式从 GPT 生成的文本中提取问答对。
    
    参数：
        text (str): GPT 返回的文本，包含多个 [问题, 答案] 格式的问答对
    
    返回：
        list: 解析出的问答对列表，每个元素为 [问题, 答案] 的列表
    """
    # 定义正则表达式匹配：[ "问题", "答案" ]
    pattern = r'\[\s*(.*?)\s*,\s*(.*?)\s*\]'
    
    # 使用 re.findall 提取所有匹配的问答对
    matches = re.findall(pattern, text)
    
    # 结果转换成标准问答格式
    qa_pairs = [[q.strip(), a.strip()] for q, a in matches]

    return qa_pairs

def get_keywords(text):
    """
    使用正则表达式从大模型返回的文本中提取关键词及其类别，返回格式：[["糖尿病", "head", "病因", "relation", "", "tail"], [...]]
    """
    # 匹配 [['关键词', '类型'], ['关键词', '类型'], ...]
    pattern = r"\[\s*\['(.*?)',\s*'(\w+)'\](?:,\s*\['(.*?)',\s*'(\w+)'\])?(?:,\s*\['(.*?)',\s*'(\w+)'\])?\s*\]"

    # 找到所有匹配项
    matches = re.findall(pattern, text)

    # 处理结果，确保返回 ['xxx', 'head', 'xxx', 'relation', 'xxx', 'tail'] 的格式
    extracted_keywords = []
    for match in matches:
        extracted_keywords.append([match[0], match[1], match[2], match[3], match[4], match[5]])

    return extracted_keywords

def get_answer_and_triples(text):
    # 提取答案部分
    answer_match = re.search(r'答案\s*:\s*(.*?)\s*/\s*使用到的三元组', text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None

    # 提取所有完整三元组 "(..., ..., ...)"，确保只提取括号包裹的三元组
    triple_matches = re.findall(r'\(([^()]+?,[^()]+?,[^()]+?)\)', text)
    used_triples = []
    for match in triple_matches:
        parts = [item.strip() for item in match.split(',', maxsplit=2)]
        if len(parts) == 3:
            used_triples.append(parts)

    return answer, used_triples

def get_chunk_id(result):
    #transfer rerank string into chunk_id list
    matches = re.findall(r'\[.*?\]', result)
    #import pdb; pdb.set_trace()
    #result_array = ast.literal_eval(matches)
    chunk_id_list = json.loads(matches[0])
    return chunk_id_list
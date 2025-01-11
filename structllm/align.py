import re
import structllm as sllm
import ast
from sentence_transformers import SentenceTransformer,util
import torch
import re

    
class SentenceBertRetriever:
    def __init__(self, corpus) -> None:

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        # self.device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
        print(f"retrieve device:{self.device}")
        self.retrieve_model = SentenceTransformer(
            'xiaobu-embedding-v2',
            device=self.device
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
    if args.retriever_align == "text-embedding-ada-002":
        retriever = OpenAIRetriever(corpus)
    elif args.retriever_align == "SentenceBERT":
        retriever = SentenceBertRetriever(corpus)
    elif args.retriever_align == "GTE":
        retriever = GTERetriever(corpus)
    else:
        pass
    return retriever


def get_target_type(args, response, cgdata):
    pattern = r'\((.*?)\)'  # 正则表达式模式，匹配括号中的内容
    #pattern = r'\{(.*?)\}'
    matches = re.findall(pattern, response)  # 查找所有匹配的内容
    result = ''
    list_rel = []
    if matches:
        result = matches[0]  # 获取第一个匹配的内容
        result = match = re.search(r"[\u4e00-\u9fff]+", result)[0]
        print(result)
        _, CG_relations = get_entitise_relations(cgdata)
        retriever = GetRetriever(args, CG_relations)
        top1_api = retriever.get_topk_candidates(1,result)
        label_rel = [CG_relations[char[0]] for char in top1_api]
    else:
        top1_api = None
        label_rel = None
    return top1_api, label_rel

def get_qa_parameter(args, text):
    list_qa = []
    #text = "{(Person1:question1,Person2:answer1),(John:How are you?,Alice:I'm fine),(Tom:What time is it?,Mary:It's 3 PM)}"
    # 使用正则表达式提取每一对问答
    pattern = r"\(([^:]+:[^,]+),([^:]+:[^,]+)\)"
    matches = re.findall(pattern, text)

    # 初始化两个数组
    questions = []
    answers = []

    # 提取每一对问题和答案
    for match in matches:
      # 提取问题和答案（通过 `:` 分隔，获取问题和答案的内容）
        question = match[0].split(":")[1]  # 获取 question 部分（去除人员名称）
        answer = match[1].split(":")[1]  # 获取 answer 部分（去除人员名称）
        # 将问题和答案添加到对应的数组中
        questions.append(question)
        answers.append(answer)

    # 将问题和答案结合为字典并存入 list_qa
    list_qa = [{"q": q, "a": a} for q, a in zip(questions, answers)]
    # 输出最终结果
    print(list_qa)
    return list_qa

def update_head_entity(args, head_entity, CG_relations):
    retriever = GetRetriever(args, CG_relations)
    # retriever = OpenAIRetriever(CG_relations)
    top5_api = retriever.get_topk_candidates(5,head_entity)
    label_rel = [CG_relations[char] for char in top5_api[0]]
    retrieve_prompt = sllm.prompt.retrieve_prompt(head_entity, label_rel, CG_relations)
    llm = sllm.llm.gpt(args)
    response = llm.get_response(retrieve_prompt.naive_prompt)
    # cleaned_string = response.lstrip().replace("'", "") # 删除所有单引号'和前置空格
    # cleaned_string = response.strip().replace("'", "").replace('"', "")
    res = label_rel[0] # 默认为第一个
    for relation in CG_relations: # 寻找lable relation
        if response.find(relation) != -1:
            res = relation
            break
    return "key='"+res+"', value"

def text2query(args, text, question, cgdata):
    """translate text into formal queries
    output: query class
    """
    pattern_step = r"(Step.*)"
    step_text = re.findall(pattern_step, text) # 得到list形式的query
    print(f"step_text:{step_text}")

    pattern_query = r'Query\d+\s*:\s*"(.*?)"'
    text = re.findall(pattern_query, text) # 得到list形式的query
    print(f"text:{text}")

    CG_entities, CG_relations = get_entitise_relations(cgdata)

    # update head entity into key-value pair
    if text[0].find("head_entity") != -1:
        head_entity = re.findall(r'head_entity=\'([^"]*)\'', text[0])
        text[0] = text[0].replace("head_entity", update_head_entity(args, head_entity[0], CG_relations))
    
    # update relation with GPT
    # text = get_relation_alignment(args, text, question, CG_relations)
    # print(text)
    
    Re_entities, Re_relations = get_parameters(text) #参数提取
    print(f"Re_entities:{Re_entities}")
    print(f"Re_relations:{Re_relations}")
    # retrieve entities
    if Re_entities == []:
        label_ent = []
    else:
        retriever = GetRetriever(args, CG_entities)
        top1_api = retriever.get_topk_candidates(1,Re_entities)
        label_ent = [CG_entities[char[0]] for char in top1_api]
    
    # retrieve relations
    if Re_relations == []:
        label_rel = []
    else:
        # retriever = OpenAIRetriever(CG_relations)
        retriever = GetRetriever(args, CG_relations)
        top1_api = retriever.get_topk_candidates(1,Re_relations)
        label_rel = [CG_relations[char[0]] for char in top1_api]

    text_query, id_query = replace_query(text, Re_entities, label_ent, Re_relations, label_rel, cgdata.node2id)
    return text_query, id_query, step_text

def replace_query(responses, Re_entities, label_ent, Re_relations, label_rel, node2id):
    # 创建映射字典
    Re_list = Re_entities + Re_relations
    label_list = label_ent + label_rel
    replace_dict = dict(zip(Re_list, label_list))
    # 使用正则表达式查找要替换的元素
    pattern = '|'.join(re.escape("'"+key+"'") for key in Re_list)
    def replace_match(match):
        matched_text = match.group()
        element = matched_text[1:-1]  # 去掉单引号
        replacement = replace_dict.get(element, element)
        return f"'{replacement}'"

    # text_query = [re.sub(pattern, lambda x: replace_dict[x.group()], response) for response in responses]
    text_query = [re.sub(pattern, replace_match, response) for response in responses]
    # text_query = re.sub(pattern, lambda x: replace_dict[x.group()], responses)
    # 创建映射字典
    Re_list = label_list
    label_list = [ node2id[node] for node in Re_list]
    #import pdb; pdb.set_trace();
    replace_dict = dict(zip(Re_list, label_list))
    # 使用正则表达式查找要替换的元素
    pattern = '|'.join(re.escape("'"+key+"'") for key in Re_list)
    # id_query = [re.sub(pattern, lambda x: replace_dict[x.group()], response) for response in text_query]
    id_query = [re.sub(pattern, replace_match, response) for response in text_query]
    print("5")
    return text_query, id_query

# def extract_query(text):
#     function = None 
#     pass 

def getStr(x): return x if type(x)==str else str(x)

def get_parameters(responce_from_api):
    entities = []
    relations = []
    # pattern = r"'(.*?)'"
    rel_pattern = r"(?:relation|key)\s*=\s*'(.*?)'(?:,|\))"
    ent_pattern = r"(?:entity|value)\s*=\s*'(.*?)'(?:,|\))"

    for item in responce_from_api:
        matches_ent = re.findall(ent_pattern, item)
        matches_rel = re.findall(rel_pattern, item)
        for match in matches_ent:
            if match[:6] != 'output' and getStr(match) not in entities and getStr(match) != '':
                entities.append(getStr(match))
        for match in matches_rel:
            if match[:6] != 'output' and getStr(match) not in relations and getStr(match) != '':
                relations.append(getStr(match))

    return entities, relations

def get_entitise_relations(cgdata):
    entities = set()
    relations = set()
    for h,r,t in cgdata.triples:
        if h == 'row_number': continue
        if t == '[0]':
            relations.add(getStr(r))
            entities.add(getStr(h))
        else:
            entities.add(getStr(r))
            entities.add(getStr(t))
    
    CG_entities = [ item for item in list(entities) if item[0:6] != '[line_' and item!='']

    return CG_entities, list(relations)


async def MetaQA_text2query(args, text, question, cgdata, CG_relations):
    """translate text into formal queries
    output: query class
    """
    pattern_step = r"(Step.*)"
    step_text = re.findall(pattern_step, text) # 得到list形式的step
    print(f"step_text:{step_text}")

    # pattern_query = r'Query\d+\s*:\s*"(.*?)"'
    # text = re.findall(pattern_query, text) # 得到list形式的query
    # print(f"text:{text}")

    pattern_query = r'Query\d+\s*:\s*"(.*?)"(\n|$)'
    tmp_text = re.findall(pattern_query, text) # 得到list形式的query
    text = []
    for item in tmp_text:
        text.append(item[0].replace('"', "'"))
    print(f"text:{text}")

    # if CG_relations == None:
    CG_entities, CG_relations = get_entitise_relations(cgdata)
    
    Re_entities, Re_relations = get_parameters(text) #参数提取
    print(f"Re_entities:{Re_entities}")
    print(f"Re_relations:{Re_relations}")

    if "ccks" in args.folder_path.lower():
        if Re_relations == []:
            label_rel = []
        else:
            results_relation = await sllm.retrieve.get_align_collection_and_query(retriever = args.retriever_align, name = "main",chroma_dir= args.chroma_dir ,query_texts = Re_relations , recall_num= 1)
            label_rel = [candidate_question[0] for candidate_question in results_relation['documents']]
    else:
        # retrieve relations
        if Re_relations == []:
            label_rel = []
        else:
            retriever = GetRetriever(args, CG_relations)
            top1_api = retriever.get_topk_candidates(1,Re_relations)
            label_rel = [CG_relations[char[0]] for char in top1_api]
    
    print("here")
    text_query, id_query = replace_query(text, Re_entities, Re_entities, Re_relations, label_rel, cgdata.node2id)
    
    return text_query, id_query, step_text

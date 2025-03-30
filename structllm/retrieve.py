import os
import chromadb
import torch
from torch import Tensor
from transformers import logging
from tqdm.autonotebook import trange
from typing import List, Union, TypeVar, Dict
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Images
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from chromadbx import NanoIDGenerator
logging.set_verbosity_error()
Embeddable = Union[Documents, Images]
D = TypeVar("D", bound=Embeddable, contravariant=True)
model = None

def get_model(model_name):
    global model
    if model is None:
        model = SentenceTransformer(
            model_name_or_path = model_name,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
    return model

async def get_embedding_BERT(text) -> List[float]:
    """ 用于对文本进行编码,编码器是MiniCPM-Embedding-Light的,返回值是float list """
    
    """
    global retrieve_model
    retrieve_model = SentenceTransformer(
        model_name_or_path=model,
        device = "cuda:0" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    """
    retrieve_model = get_model("MiniCPM-Embedding-Light")
    
    if isinstance(text, str):
        result = retrieve_model.encode([text])
        flat_array = result.tolist()
        return flat_array
    elif isinstance(text, list):
        result = retrieve_model.encode(text)
        flat_array = result.tolist()
        return flat_array

class NewEmbeddingFunction(EmbeddingFunction):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def __call__(self, input: D) -> Embeddings:
        embeddings = self.encoder.encode(input)
        return embeddings

class EncoderBERT:
    async def encode(
        self,
        text: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = False,
        **kwargs
    ) -> List[Tensor]:
        text_embeddings = []
        for batch_start in trange(
            0, len(text), batch_size, disable=show_progress_bar
        ):
            batch_end = batch_start + batch_size
            batch_text = text[batch_start:batch_end]
            #print(f"Batch {batch_start} to {batch_end-1}")
            assert "" not in batch_text
            resp = await get_embedding_BERT(batch_text)
            """for i, be in enumerate(resp):
                assert (
                        i == be.index
                )  # double check embeddings are in same order as input
                """
            batch_text_embeddings = [e for e in resp]
            text_embeddings.extend(batch_text_embeddings)
        return text_embeddings

class BERT:
    def __init__(self) -> None:
        self.q_model = EncoderBERT()

    async def encode(
        self, queries: List[str], batch_size: int = 16, model=None, **kwargs
    ) -> List[Tensor]:
        return await self.q_model.encode(queries, batch_size=batch_size, model=model,**kwargs)

class Encoder:
    def __init__(self, encoder_name: str) -> None:
        self.encoder_name = encoder_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if encoder_name == "SentenceBERT":
            self.encoder = BERT()
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                "MiniCPM-Embedding-Light",
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
            )
        else :pass

async def _get_embedding_and_save_to_chroma(
    data: List[Dict[str, str]],
    collection: Collection,
    encoder: Encoder,
    batch_size: int = 64,
):
    """ 将编码存入chroma向量数据库 """
    encoder_ = encoder.encoder

    docs = [item["question"] for item in data] #question list
    meta_keys = list(data[0].keys())           #data中dict的keys
    del meta_keys[meta_keys.index("question")] #删除question，只剩下content
    embeddings = await encoder_.encode(docs, batch_size=batch_size, show_progress_bar=True, model=model)
    # else:
    #     embeddings = encoder_.doc_model.encode(
    #         docs, batch_size=batch_size, show_progress_bar=True
    #     )
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()

    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in range(len(docs))],
        )
    return collection

#prompt存储
def put_embedding_prompt(retriever: str, prompt_data: list = None, collection = None):

    encoder = Encoder(retriever,model=model)
    retrieve_data = []

    for idx , elements in enumerate(prompt_data):
        k,v = elements[0],elements[1]
        data_prompt = {
                    "question": k,
                    "content": v
        }
        if(data_prompt in retrieve_data): continue
        retrieve_data.append(data_prompt)
    _get_embedding_and_save_to_chroma(retrieve_data, collection, encoder)
    return collection

#获取collection
async def get_collection(name: str = None, encoder = None):

    encoder = encoder
    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    Collection =  await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    return Collection

#获取qas_collection并存储prompt
async def get_qas_collection_and_write(qa_data: list = None, name: str = None , chunk_id = None, encoder = None):
#1 起服务
    encoder = encoder
    if name == None:
        name = "qas"
    
    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
#2 拆包
    retrieve_data = []

    for idx , elements in enumerate(qa_data):
        n,q,ans = elements
        data_prompt = {
                    "name": n,
                    "question": q,
                    "answer": ans,
                    "chunk_id":chunk_id
        }
        if(data_prompt in retrieve_data): continue
        retrieve_data.append(data_prompt)
#3 编码，存储
    encoder_ = encoder.encoder
    docs = [item["question"] for item in retrieve_data] #question list
    meta_keys = list(retrieve_data[0].keys())           #data中dict的keys
    #del meta_keys[meta_keys.index("question")] #删除question，只剩下content
    embeddings = encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
 
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            await collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: retrieve_data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        await collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: retrieve_data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )
    return None

#获取qas_collection并查询top K
def get_qas_collection_and_query(name: str = None, query_texts: str = None ,recall_num : int = None, encoder = None):
    
    encoder = encoder
    if name == None:
        name = "qas"
    
    embedding_function = encoder.ef
    chroma_client =  chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    results_qas = collection.query(query_texts=query_texts , n_results = recall_num)
    return results_qas

#数据库记录数量
async def chroma_count(name: str = None , encoder = None):
    encoder = encoder
    if name == None:
        name = "main"
    
    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    num = collection.count()
    return num

#重置数据库
async def rebuild_collection(name: str = None , encoder=None):
    
    encoder = encoder
    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    await chroma_client.delete_collection(name)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )

    return collection

#获取context_collection并查询top K
async def get_context_collection_and_query(name: str = None, query_texts: str = None, recall_num : int = None, encoder=None):
    
    encoder = encoder
    if name == None:
        name = "context"
    
    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    results_context = collection.query(query_texts = query_texts, n_results = recall_num)
    return results_context

#获取summary_collection并查询top K
async def get_summary_collection_and_query(retriever: str, name: str = None, chroma_dir: str = None, query_texts: str = None, recall_num : int = None, model=None):
    
    encoder = Encoder(retriever,model=model)
    if name == None:
        name = "summary"
    
    embedding_function = encoder.ef
    chroma_client =  chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    results_summary = collection.query(query_texts = query_texts, n_results = recall_num)
    return results_summary

#获取qa_collection并存储prompt
async def get_context_collection_and_write(retriever: str, context_data: list = None, name: str = None, chroma_dir: str = None, chunk_id = None, model=None):
#1 起服务
    encoder = Encoder(retriever,model=model)
    if name == None:
        name = "context"
    
    embedding_function = encoder.ef
    chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
#2 拆包
    retrieve_data = []

    for i in range(len(context_data)):
        data = {
                    "context": context_data[i],
                    "chunk_id": chunk_id
        }
        if(data in retrieve_data): continue
        retrieve_data.append(data)
#3 编码，存储
    encoder_ = encoder.encoder
    docs = [item["context"] for item in retrieve_data] #question list
    meta_keys = list(retrieve_data[0].keys())           #data中dict的keys
    #del meta_keys[meta_keys.index("")] #删除question，只剩下content
    embeddings = encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
 
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: retrieve_data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: retrieve_data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )
    return None

async def get_triples_collection_and_write(triple_data: list = None, name: str = None, encoder = None):
    """
    将提取到的三元组数据写入数据库。

    参数：
        retriever: 用于初始化编码器的模型标识（例如模型名称）
        triple_data: 三元组数据列表，每个元素为形如 [a, b, c] 的三元组
        name: 数据库集合名称，如果未指定，默认为 "triples"
        chroma_dir: chromadb 的工作目录（如有需要）
    """
    encoder = encoder
    if name is None:
        name = "triples"

    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )

    # 2. 准备数据
    retrieve_data = []
    for i in range(len(triple_data)):
        data = {
            "triple": triple_data[i],  # 三元组数据
        }

        # 过滤掉元数据为 None 的项
        if any(value is None for value in data.values()):
            continue  # 如果有元数据为 None，则跳过这一条记录

        retrieve_data.append(data)

    # 将每个三元组转换为字符串（或可选的 JSON 格式），便于编码
    docs = [str(item["triple"]) for item in retrieve_data]
    meta_keys = list(retrieve_data[0].keys())  # 获取字典中的所有键

    # 3. 编码并写入数据库
    encoder_ = encoder.encoder
    embeddings = await encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()

    # 处理元数据，将列表转换为字符串（如果是列表的话）
    def process_metadata(metadata):
        if isinstance(metadata, list):  # 如果元数据是列表类型
            return " , ".join(str(x) for x in metadata)  # 使用 ',' 连接每个元素并转换为字符串
        return metadata  # 否则直接返回元数据

    # 分批添加数据，避免一次性写入过多数据
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            await collection.add(
                embeddings=embeddings[i: i + 20000],
                documents=docs[i: i + 20000],
                metadatas=[
                    {key: process_metadata(retrieve_data[j][key]) for key in meta_keys}
                    for j in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(j) for j in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        await collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: process_metadata(retrieve_data[i][key]) for key in meta_keys}
                for i in range(len(docs))
            ],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )
    return None

async def get_triples_head_collection_and_write(data: list = None, name: str = None, encoder = None):
    """
    参数：
        retriever: 用于初始化编码器的模型标识（例如模型名称）
        triple_data: 三元组数据列表，每个元素为形如 {"head": value, "relation": value, "tail": value} 的字典
        name: 数据库集合名称，如果未指定，默认为 "heads"
        chroma_dir: chromadb 的工作目录（如有需要）
    """
    encoder = encoder
    if name is None:
        name = "triple_head"  # 只存储 head 部分，集合名称更改为 "head"

    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )

    # 2. 准备数据，仅保留 head 部分
    retrieve_data = []
    for triple in data:
        if "head" in triple and triple["head"] is not None:
            retrieve_data.append({"head": triple["head"]})

    if not retrieve_data:
        print("未找到有效的 head 数据，跳过写入 ChromaDB。")
        return None

    # 仅存储 head 数据
    docs = [item["head"] for item in retrieve_data]

    # 3. 编码并写入数据库
    encoder_ = encoder.encoder
    embeddings = await encoder_.encode(docs, batch_size=64, show_progress_bar=True, model=model)
    if embeddings is None:
        raise ValueError("编码器返回None嵌入")
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
    
    def safe_get_metadata(i):
            return {
                "head": retrieve_data[i]["head"],
                "relation": retrieve_data[i].get("relation", ""),  # 防御性访问
                "tail": retrieve_data[i].get("tail", "")
            }

    # 分批添加数据，避免一次性写入过多数据
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
         await collection.add(
                embeddings=embeddings[i: i + 20000],
                documents=docs[i: i + 20000],
                metadatas=[safe_get_metadata(j) for j in range(i, min(len(embeddings), i + 20000))],
                ids=[str(j) for j in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        await collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[safe_get_metadata(j) for j in range(len(docs))],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )

    return None

async def get_triples_relation_collection_and_write(data: list = None, name: str = None, encoder = None):
    """
    仅将提取到的三元组数据中的 relation 部分写入数据库。

    参数：
        retriever: 用于初始化编码器的模型标识（例如模型名称）
        triple_data: 三元组数据列表，每个元素为形如 {"head": value, "relation": value, "tail": value} 的字典
        name: 数据库集合名称，如果未指定，默认为 "relations"
        chroma_dir: chromadb 的工作目录（如有需要）
    """
    # 1. 初始化服务与编码器
    from chromadb import HttpClient
    encoder = encoder
    if name is None:
        name = "triple_relation"  # 只存储 relation 部分，集合名称更改为 "relations"

    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )

    # 2. 准备数据，仅保留 relation 部分
    retrieve_data = []
    for triple in data:
        if "relation" in triple and triple["relation"] is not None:
            retrieve_data.append({"relation": triple["relation"]})

    if not retrieve_data:
        print("未找到有效的 relation 数据，跳过写入 ChromaDB。")
        return None

    # 仅存储 relation 数据
    docs = [item["relation"] for item in retrieve_data]

    # 3. 编码并写入数据库
    encoder_ = encoder.encoder
    embeddings = await encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if embeddings is None:
        raise ValueError("编码器返回None嵌入")
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()

    def safe_get_metadata(i):
            return {
                "head": retrieve_data[i].get("head",""),
                "relation": retrieve_data[i]["relation"],  # 防御性访问
                "tail": retrieve_data[i].get("tail", "")
            }

    # 分批添加数据，避免一次性写入过多数据
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            await collection.add(
                embeddings=embeddings[i: i + 20000],
                documents=docs[i: i + 20000],
                metadatas=[safe_get_metadata(j) for j in range(i, min(len(embeddings), i + 20000))],
                ids=[str(j) for j in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        await collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[safe_get_metadata(j) for j in range(len(docs))],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )

    return None

async def get_triples_tail_collection_and_write(data: list = None, name: str = None, encoder = None):
    """
    仅将提取到的三元组数据中的 tail 部分写入数据库。

    参数：
        retriever: 用于初始化编码器的模型标识（例如模型名称）
        triple_data: 三元组数据列表，每个元素为形如 {"head": value, "relation": value, "tail": value} 的字典
        name: 数据库集合名称，如果未指定，默认为 "tails"
        chroma_dir: chromadb 的工作目录（如有需要）
    """
    # 1. 初始化服务与编码器
    encoder = encoder
    if name is None:
        name = "triple_tail"  # 只存储 tail 部分，集合名称更改为 "tails"

    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )

    # 2. 准备数据，仅保留 tail 部分
    retrieve_data = []
    for triple in data:
        if "tail" in triple and triple["tail"] is not None:
            retrieve_data.append({"tail": triple["tail"]})

    if not retrieve_data:
        print("未找到有效的 tail 数据，跳过写入 ChromaDB。")
        return None

    # 仅存储 tail 数据
    docs = [item["tail"] for item in retrieve_data]

    # 3. 编码并写入数据库
    encoder_ = encoder.encoder
    embeddings = await encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if embeddings is None:
        raise ValueError("编码器返回None嵌入")
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
    
    def safe_get_metadata(i):
            return {
                "head": retrieve_data[i].get("head",""),
                "relation": retrieve_data[i].get("relation",""),  # 防御性访问
                "tail": retrieve_data[i]["tail"]
            }

    # 分批添加数据，避免一次性写入过多数据
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            await collection.add(
                embeddings=embeddings[i: i + 20000],
                documents=docs[i: i + 20000],
                metadatas=[safe_get_metadata(j) for j in range(i, min(len(embeddings), i + 20000))],
                ids=[str(j) for j in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        await collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[safe_get_metadata(i) for i in range(len(docs))],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )

    return None

async def get_qa_collection_and_write(data: list = None, name: str = None, encoder = None):
    """
    将问答对 (Q, A) 存入 ChromaDB。

    参数：
        retriever (str): 选择的编码器名称
        data (list): 问答对数据，格式应为 [{"question": "Q1", "answer": "A1"}, ...]
        name (str): 存储集合名称，默认 "qa_pairs"
        chroma_dir (str): ChromaDB 存储目录（可选）
    """
    encoder = encoder
    if name is None:
        name = "qa_pairs"  # 存储问答对的集合名称
    embedding_function = encoder.ef

    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )

    # 2. 准备数据，仅保留有效的 Q/A
    retrieve_data = []
    for qa in data:
        if "question" in qa and "answer" in qa and qa["question"] and qa["answer"]:
            retrieve_data.append({"question": qa["question"], "answer": qa["answer"]})

    if not retrieve_data:
        print("未找到有效的 QA 数据，跳过写入 ChromaDB。")
        return None

    # 仅存储问题部分用于向量检索
    docs = [item["question"] for item in retrieve_data]

    # 3. 编码并写入数据库
    encoder_ = encoder.encoder
    embeddings = await encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    
    if embeddings is None:
        raise ValueError("编码器返回 None 嵌入")
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()

    def safe_get_metadata(i):
        return {
            "question": retrieve_data[i]["question"],
            "answer": retrieve_data[i]["answer"]
        }

    # 分批添加数据，避免一次性写入过多数据
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            await collection.add(
                embeddings=embeddings[i: i + 20000],
                documents=docs[i: i + 20000],
                metadatas=[safe_get_metadata(j) for j in range(i, min(len(embeddings), i + 20000))],
                ids=[str(j) for j in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        await collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[safe_get_metadata(j) for j in range(len(docs))],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )

    return None

#获取qa_collection并存储prompt
async def get_summary_collection_and_write(summarydata, chunk_data, name: str = None, chunk_id = None, encoder=None):

#1 起服务
    encoder = encoder
    if name == None:
        name = "summary"
    
    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
#2 拆包
    retrieve_data = []
    data_prompt = {
        "summary": summarydata,
        "chunk_id": chunk_id,
        "chunk_data": chunk_data
    }
    retrieve_data.append(data_prompt)
#3 编码，存储
    encoder_ = encoder.encoder
    docs = [item["summary"] for item in retrieve_data] #question list
    meta_keys = list(retrieve_data[0].keys())           #data中dict的keys
    #del meta_keys[meta_keys.index("question")] #删除question，只剩下content
    embeddings = encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
 
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: retrieve_data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: retrieve_data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )
    return None

async def get_path_collection_and_write(path, encoder):
    
#1 起服务
    encoder = encoder
    name = "path"
    
    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
#2 拆包
    retrieve_data = []
    data_prompt = {
        "name": name ,
        "path": path
    }
    retrieve_data.append(data_prompt)
#3 编码，存储
    encoder_ = encoder.encoder
    docs = [item["name"] for item in retrieve_data] #question list
    meta_keys = list(retrieve_data[0].keys())           #data中dict的keys
    #del meta_keys[meta_keys.index("question")] #删除question，只剩下content
    embeddings = await encoder_.encode(docs, batch_size=64, show_progress_bar=True)
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
 
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            await collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: retrieve_data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        await collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: retrieve_data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in NanoIDGenerator(len(docs))],
        )
    return None

async def get_output_path(recall_num : int = 1, encoder=None):
    encoder = encoder
    name = "path"
    
    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    results_summary = await collection.query(query_texts = name, n_results = recall_num)
    return results_summary

async def get_summary_collection_and_query_chunk(retriever: str, chunk_id = None, model=None):
    
    encoder = Encoder(retriever,model=model)
    name = "summary"
    
    embedding_function = encoder.ef
    chroma_client = await chromadb.AsyncHttpClient(host='127.0.0.1', port=8000)
    collection = await chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    qas_result = await collection.get(where={"chunk_id": chunk_id})
    return qas_result

# -*- encoding: utf-8 -*-
from sentence_transformers import SentenceTransformer,util
import torch
import re

class SentenceBertRetriever:
    def __init__(self, corpus) -> None:

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"retrieve device:{self.device}")
        self.retrieve_model = SentenceTransformer(
            r'/home/wcy/code/InterviewSystem-v0.1/stella_en_400M_v5',
            device=self.device,
            trust_remote_code=True
        )
        self.corpus = corpus

    def get_embedding(self, text):
        result = self.retrieve_model.encode(text)
        flat_array = result.tolist()
        return flat_array

    def get_topk_candidates(self, topk, query):

        # query is a list

        # Corpus of documents and their embeddings
        api_corpus_embeddings = self.get_embedding(self.corpus)
        # import pdb; pdb.set_trace();
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


corpus = ['执导', '前夫', '叔外祖父', '连载平台', '弟媳', '前妻', '姑父', '妯娌', '毕业院校', '所属机构', '亲家公', '未婚妻', '堂兄', '专职院士数', '前女友', '舅母', '配音', '外孙女', '表姑父', '叔叔', '岳母', '师兄', '女朋友', '哥哥', '师妹', '搭档', '庶子', '丈夫', '经纪人', '第二任妻子', '前公公', '现任领导', '妹', '知己', '养母', '孙女', '导师', '曾外祖父', '玄孙', '第六任妻子', '文学作品', '音乐作品', '亲家母', '义弟', '第一任丈夫', '制作', '综艺节目', '师弟', '表妹', '伯母', '小叔子', '前儿媳', '养父', '办学性质', '战友', '母亲', '义女', '摄影作品', '音乐视频', '叔外公', '合作人', '连襟', '参演', '外甥女婿', '学长', '大舅哥', '岳父', '云孙', '义子', '师爷', '姑妈', '徒弟', '院系设置', '先夫', '亡妻', '义父', '侄女', '姐姐', '师生', '学校特色', '主要角色', '生母', '公公', '出版社', '学妹', '主持', '曾外孙子', '编剧', '义母', '经纪公司', '表姐', '简称', '第一任妻子', '发行专辑', '小姨子', '婶母', '伯乐', '外曾祖父', '姨夫', '祖母', '曾孙子', '历任领导', '原配', '大舅子', '外祖母', '偶像', '堂弟', '外曾祖母', '学弟', '妻姐', '对手', '堂小舅子', '生父', '师姐', '祖父', '孙子', '叔父', '表侄', '员工', '第五任妻子', '养子', '奶奶', '姑姑', '表兄', '第二任丈夫', '姐夫', '合作院校', '成员', '伴侣', '男友', '义妹', '学生','相关国内联盟', '代表作品', '爱人', '大伯哥', '教练', '类型', '父亲', '继子', '师父', '其他关系', '恋人', '兄弟', '学校身份', '继任', '队友', '男朋友', '儿媳', '堂哥', '继母', '社长', '师祖', '弟子', '堂姐', '为他人创作音乐', '主要配音', '继女', '师傅', '助理', '好友', '侄子', '类别', '表姨', '小舅子', '堂父', '姨父', '侄孙媳妇', '大姨子', '曾祖父', '外孙子', '儿子', '未婚夫', '恩师', '妾', '创始人', '代表', '妹夫', '妻子', '外甥女', '养女', '歌曲原唱', '同学', '法人', '嫂子', '老板', '伯伯', '旗下艺人', '学校类别', '伯父', '侄孙', '旧爱', '曾孙女', '第四任妻子', '舅父', '外曾孙子', '大姑子', '堂舅', '姑母母', '领导', '前队友', '大爷爷', '学姐', '继父', '外孙', '主要作品', '女儿', '前任', '外祖父', '知名人物', '曾孙', '同门', '外曾孙女', '创办', '义兄', '设立单位', '表哥', '朋友', '登场作品', '堂侄', '表弟', '姨母', '办学团体', '嫡母', '曾祖母', '表叔', '弟弟', '第三任妻子', '侄孙子', '婆婆', '曾外祖母', '女婿婿', '堂妹', '外甥', '前男友', '主要演员','挚爱', '作者', '小姑子', '老师' ]
response = "(relation='侄子', head_entity=张说)"
pattern = r'\((.*?)\)'  # 正则表达式模式，匹配括号中的内容
matches = re.findall(pattern, response)  # 查找所有匹配的内容\


print(matches[0])
if matches:
        pattern = re.compile("[\u4e00-\u9fa5]")
        matches = re.findall(pattern,matches[0])
        result = matches[0]  # 获取第一个匹配的内容
        CG_relations = corpus
        print(type(CG_relations))
        retriever = SentenceBertRetriever(CG_relations)
        # retriever = OpenAIRetriever(CG_relations)
        top1_api = retriever.get_topk_candidates(1,result)
        label_rel = [CG_relations[char[0]] for char in top1_api]
        print(top1_api)
        print(label_rel)
else:
        top1_api = None
        label_rel = None
        print(top1_api)

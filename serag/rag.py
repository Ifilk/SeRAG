from FlagEmbedding import FlagAutoModel
import faiss

from serag.utils import split_text_file

# TODO 使用文件夹
file_path = 'test.txt'
query = 'Test query'
chunk_size = 200
overlap = 20
k = 4

# 加载BGE模型
model = FlagAutoModel.from_finetuned(r'D:\Project\langchain\model\bge-small-zh-v1.5',
                                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                      use_fp16=True)
# BGE 输出向量维度为 512
dim = 512
# 建立Inner product索引
index = faiss.IndexFlatIP(dim)

chunks = split_text_file(file_path, chunk_size, overlap)

for chunk in chunks:
    feature = model.encode(chunks)
    index.add(feature)

D, I = index.search(query, k)
print(I)
# TODO 从搜索结果获取原文
# TODO 合成Prompt
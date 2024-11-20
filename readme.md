# Semantic Entropy Retrieval Augmented Generation

## 1. 依赖安装
```shell
pip install -U FlagEmbedding ragas colorlog dashscope python-dotenv faiss-cpu
```

## 计算向量相似度
```python
from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned(r'D:\Project\langchain\model\bge-small-zh-v1.5',
                                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                      use_fp16=True)
sentences_1 = ["I love NLP", "I love machine learning"]
sentences_2 = ["I love BGE", "I love text retrieval"]
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```

## Faiss 检索
```python
import faiss
import numpy as np
dim = 2048
index = faiss.IndexFlatIP(dim) # 建立Inner product索引

feature = np.random.random((1, 2048)).astype('float32')
index.add(feature)

k = 4  # 4 nearest neighbors
test = feature = np.random.random((1, 2048)).astype('float32')
D, I = index.search(test, k) # (D.shape = test.shape[0] * k, I同理)
print(I)
```
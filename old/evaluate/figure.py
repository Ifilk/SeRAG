import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV数据
data = pd.read_csv('evaluation/2024-09-30-17-57-52_qwen-turbo_bge_WikiEval.csv')

# 创建箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['answer_relevancy', 'context_precision', 'faithfulness', 'context_recall']])
plt.title('Naive RAG')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figure/naive_rag.png')

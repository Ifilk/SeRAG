import numpy as np
import pandas as pd
from renumics import spotlight
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from old.embed import load_embeddings_faiss
from old.main import ask_and_get_answer_from_local

vector_db_path = '../vector_db_backup/WikiEval_Faiss'
top_k = 3
question = 'When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?'

if __name__ == '__main__':
    db = load_embeddings_faiss(vector_db_path, 'bge')

    answer = ask_and_get_answer_from_local(model_name="qwen-turbo",
                                             vector_db=db, prompt=question,
                                             top_k=top_k)[0]

    vs = db.__dict__.get("docstore")
    index_list = db.__dict__.get("index_to_docstore_id").values()
    doc_cnt = db.index.ntotal

    # 向量空间的近似重建
    embeddings_vec = db.index.reconstruct_n()

    doc_list = list()
    for i, doc_id in enumerate(index_list):
        a_doc = vs.search(doc_id)
        doc_list.append([doc_id, a_doc.metadata.get("source"), a_doc.page_content, embeddings_vec[i]])

    df = pd.DataFrame(doc_list, columns=['id', 'metadata', 'document', 'embedding'])

    # 添加问题和答案的行
    embeddings_model = HuggingFaceBgeEmbeddings(model_name=r'D:\Project\langchain\model\bge-small-zh-v1.5',
                                                model_kwargs={'device': 'cpu'})
    question_df = pd.DataFrame(
        {
            "id": "question",
            "question": question,
            "embedding": [embeddings_model.embed_query(question)],
        })
    answer_df = pd.DataFrame(
        {
            "id": "answer",
            "answer": answer,
            "embedding": [embeddings_model.embed_query(answer)],
        })
    df = pd.concat([question_df, answer_df, df])

    question_embedding = embeddings_model.embed_query(question)
    # 添加向量距离列
    df["dist"] = df.apply(
        lambda row: np.linalg.norm(
            np.array(row["embedding"]) - question_embedding
        ), axis=1,)

    spotlight.show(df)







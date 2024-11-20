import os

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS


def get_embedding(embedding_name):
    """
    根据embedding名称去加载embedding模型
    :param embedding_name: 路径或者名称
    :return:
    """
    if embedding_name == "bge":
        embedding_path = os.environ[embedding_name]
        model_kwargs = {'device': 'cpu'}
        return HuggingFaceBgeEmbeddings(model_name=embedding_path, model_kwargs=model_kwargs)
    if embedding_name == "bce":
        return None


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings_chroma(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store


def create_embeddings_faiss(vector_db_path, embedding_name, chunks):
    """
    使用FAISS向量数据库，并保存
    :param vector_db_path: 向量
    :param embedding_name:
    :param chunks:
    :return:
    """
    embeddings = get_embedding(embedding_name)
    db = FAISS.from_documents(chunks, embeddings)

    if not os.path.isdir(vector_db_path):
        os.mkdir(vector_db_path)

    db.save_local(folder_path=vector_db_path)
    return db


def load_embeddings_faiss(vector_db_path, embedding_name):
    """
    加载向量库
    :param vector_db_path:
    :param embedding_name:
    :return:
    """
    embeddings = get_embedding(embedding_name)
    if os.path.exists(os.path.join(vector_db_path, 'index.faiss')):
        db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        return db



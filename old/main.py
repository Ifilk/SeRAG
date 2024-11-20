from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

from language_model import AlibabaLLM, Local, AlibabaModelName
from evaluate.evaluation_rag_strategy import rag_handler
from main_logger import logger


@rag_handler('naive')
def ask_and_get_answer_from_local(model_name, vector_db, prompt, top_k=5):
    """
    从本地加载大模型
    :param model_name: 模型名称
    :param vector_db:
    :param prompt:
    :param top_k:
    :return:
    """
    # docs_and_scores = vector_db.similarity_search_with_score(prompt, k=top_k)
    # logger.info("docs_and_scores: ", docs_and_scores)
    # knowledge = [doc.page_content for doc in docs_and_scores]
    # logger.debug("检索到的知识：", knowledge)
    if model_name == "qwen-turbo":
        llm = AlibabaLLM(model_name=AlibabaModelName.qwen_turbo)
    if model_name == "local":
        llm = Local()
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=DEFAULT_TEMPLATE)
    retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': top_k})
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs={"prompt": prompt_template},
                                        return_source_documents=True)
    answer = chain({"query": prompt, "top_k": top_k})
    logger.debug(f"answers: {answer}")
    # answer = chain.run(prompt)
    # answer = answer['choices'][0]['message']['content']
    answer = answer['result']
    return answer, retriever
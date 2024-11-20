
RAG_HANDLER_DICT = {}


def rag_handler(name):
    """
    fun args:
    - model_name: bool,
    - vector_db: List[str],
    - prompt: os.PathLike
    - top_k

    :param name: name of llm
    """

    def wrapper(func):
        RAG_HANDLER_DICT[name] = func
        return func

    return wrapper
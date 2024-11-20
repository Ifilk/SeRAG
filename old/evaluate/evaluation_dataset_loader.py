import os

from tqdm import tqdm
from typing_extensions import List

from datasets import load_dataset
from old.main_logger import logger
from old.util import fetch_wikipedia_content

DATASET_LOADER_DICT = {}


def dataset_loader(name):
    """
    fun args:
    - supplement: bool,
    - excludes: List[str],
    - supplement_output_path: os.PathLike

    :param name: name of dataset
    """

    def wrapper(func):
        DATASET_LOADER_DICT[name] = func

    return wrapper


@dataset_loader("explodinggradients/WikiEval")
def load(supplement: bool,
         excludes: List[str],
         supplement_output_path: os.PathLike):
    """
    question: a question that can be answered from the given Wikipedia page (source).
    source: The source Wikipedia page from which the question and context are generated.
    grounded_answer: answer grounded on context_v1
    ungrounded_answer: answer generated without context_v1
    poor_answer: answer with poor relevancy compared to grounded_answer and ungrounded_answer
    context_v1: Ideal context to answer the given question
    contetx_v2: context that contains redundant information compared to context_v1

    :param supplement:
    :param excludes:
    :param supplement_output_path:
    :return:
    """
    dataset = load_dataset("explodinggradients/WikiEval")
    s_excludes = []

    file_list = []
    if supplement:
        logger.info(f'从wikipedia上获取数据')
        for source in tqdm(dataset['train']['source']):
            path = fetch_wikipedia_content(source, supplement_output_path)
            if path is not None:
                file_list.append(os.path.join(supplement_output_path, path))
            else:
                logger.warning(f'无法从wikipedia中获取到{source}，{source}将会被忽略')
                s_excludes.append(source)

    dataset = dataset['train'].filter(lambda row: row['source'] not in excludes)
    logger.debug(f'excluded: {excludes + s_excludes}')

    total_queries = dataset["question"]
    ground_truth = dataset["context_v1"]

    return total_queries, ground_truth, file_list, s_excludes

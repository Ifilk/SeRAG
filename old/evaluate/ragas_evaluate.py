import argparse
import json
import logging
import os
from datetime import datetime
from enum import Enum

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from torch.cuda import is_available
from tqdm import tqdm

import evaluation_dataset_loader
from old.language_model import AlibabaModelName, AlibabaLLM, Local
from old.embed import create_embeddings_faiss, create_embeddings_chroma, load_embeddings_faiss
from old.evaluate import evaluation_rag_strategy
from old.main_logger import logger
from perprocess import load_document, chunk_data

class ModelType(Enum):
    qwen_turbo = 'qwen-turbo'
    local = 'local'

BASE_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser()

parser.add_argument('--v', '--verbose', action='store_true',
                    help='Verbose mode')
parser.add_argument('--auto_supplement', action='store_true',
                    help='Complete documents automatically')
parser.add_argument('--save_database', action='store_true',
                    help='Save generated database to disk')
parser.add_argument('--s_out', '--supplement_output', type=str, default='documents',
                    help='Auto supplement output directory')
parser.add_argument('-model_name', type=str, required=True,
                    help='Model name')
parser.add_argument('-rag_strategy', type=str, required=True,
                    help='Strategy of rag')
parser.add_argument('-embedding_model', type=str, required=True,
                    help='Embedding model name')
parser.add_argument('-vector_db', type=str, default='FAISS',
                    help='Support of FAISS and Chroma')
parser.add_argument('-load_index', type=str,
                    help='Path of FAISS files')
parser.add_argument('-chunk_size', type=int, default=512,
                    help='Chunk size')
parser.add_argument('-top_k', type=int, default=3,
                    help='Top k')
parser.add_argument('-chunk_overlap', type=int, default=150,
                    help='Chunk overlap')
parser.add_argument('-d', '-document', type=str, dest='documents', action='append',
                    help='Path of document to evaluate')
parser.add_argument('-ds', '-documents_dir', type=str, dest='documents_dirs', action='append',
                    help='Path of document directories')
parser.add_argument('-e', '-exclude', type=str, dest='excludes', action='append',
                    help='Source name of excluded data row')
parser.add_argument('-dataset', type=str, dest='dataset', required=True,
                    help='Path of evaluation datasets')

args = parser.parse_args()
SAVE_FILE = "_intermediate_results.json"

if args.v:
    logger.setLevel(logging.DEBUG)

try:
    model_name = ModelType[args.model_name]
    embedding_model = args.embedding_model
    chunk_size = args.chunk_size
    top_k = args.top_k
    chunk_overlap = args.chunk_overlap
    file_list = args.documents if args.documents is not None else []
    excludes = args.excludes if args.excludes is not None else []

    dataset = None
    if args.dataset and os.path.exists(args.dataset):
        dataset = Dataset.load_from_disk(args.dataset)
        logger.info(f'已加载{len(dataset)}条数据，从{args.dataset}')

    if not dataset:
        vector_db_path = os.path.join(BASE_DIR, "vector_db")
        # dataset = load_dataset("explodinggradients/WikiEval")
        # if args.s:
        #     logger.info(f'从wikipedia上获取数据')
        #     for source in tqdm(dataset['train']['source']):
        #         path = fetch_wikipedia_content(source, args.s_out)
        #         if path is not None:
        #             file_list.append(path)
        #         else:
        #             logger.warning(f'无法从wikipedia中获取到{source}，{source}将会被忽略')
        #             excludes.append(source)
        #
        # for fd in os.listdir(args.s_out):
        #     file_name = os.path.join(fd)
        #     file_list.append(os.path.join(BASE_DIR, args.s_out, fd))
        #
        # dataset = dataset['train'].filter(lambda row: row['source'] not in excludes)
        # logger.debug(f'excluded: {excludes}')
        try:
            total_queries, ground_truth, s_document, s_excludes =\
                evaluation_dataset_loader.DATASET_LOADER_DICT[args.dataset](args.auto_supplement, excludes, args.s_out)
        except KeyError:
            logger.error(f'找不到数据集{dataset}')
            exit(1)

        file_list += [os.path.join(BASE_DIR, args.s_out, fd) for fd in s_document]
        excludes += s_excludes

        documents_dirs = args.documents_dirs if args.documents_dirs is not None else []
        for ds in documents_dirs:
            for fd in os.listdir(ds):
                file_list.append(os.path.join(BASE_DIR, args.s_out, fd))

        chunks = []

        for file_name in file_list:
            data = load_document(file_name)
            chunks += chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if args.load_index is None:
            logger.info(f'创建{args.vector_db}向量数据库...')
            if args.vector_db == "Chroma":
                vector_store = create_embeddings_chroma(chunks)
            else:
                if args.vector_db != "FAISS":
                    logger.warn(f'Unsupported vector database {args.vector_db}, using FAISS')
                vector_store = create_embeddings_faiss(vector_db_path=vector_db_path, embedding_name="bge",
                                                       chunks=chunks)
        else:
            vector_store = load_embeddings_faiss(args.load_index, 'bge')

        answers = []
        contexts = []
        processed_queries = []

        if os.path.exists(SAVE_FILE):
            logger.info('检测到有先前保存的进度，是否加载(y/n):')
            reply = input()
            if reply == 'y':
                with open(SAVE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"加载了之前保存的进度：{SAVE_FILE}")
                    answers, contexts, processed_queries = data["answers"], data["contexts"], data["processed_queries"]

        logger.info(f'使用{model_name.value}生成回答')

        remaining_queries = [q for q in total_queries if q not in processed_queries] \
            if len(processed_queries) > 0 else total_queries

        for query in tqdm(remaining_queries):
            try:
                answer, retriever = evaluation_rag_strategy.RAG_HANDLER_DICT[args.rag_strategy](model_name=model_name.value,
                                                                                                vector_db=vector_store,
                                                                                                prompt=query,
                                                                                                top_k=top_k)
                answers.append(answer)
                processed_queries.append(query)
                contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])
            except KeyError as e:
                logger.error(f'无法找到rag策略{args.rag_strategy}')
                raise e
            except Exception as e:
                logger.info(f'模型在回答{query}时发生错误{str(e)}，正在保存数据')
                data = {
                    "answers": answers,
                    "contexts": contexts,
                    "processed_queries": processed_queries
                }
                with open(SAVE_FILE, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
                raise e

        data_samples = {
            'question': total_queries,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': [g[0] for g in ground_truth][0]
        }
        try:
            dataset = Dataset.from_dict(data_samples)
            if args.save_database:
                time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                path = f'datasets/dataset_{time_stamp}'
                dataset.save_to_disk(path)
                logger.info(f'生成的数据库已经保存到{path}')
        except Exception as e:
            logger.info(f'在生成数据库时发生错误{str(e)}，正在保存数据')
            data = {
                "answers": answers,
                "contexts": contexts,
                "processed_queries": processed_queries
            }
            with open(SAVE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            raise e

    logger.info(f'开始生成评估')
    if model_name == ModelType.qwen_turbo:
        llm = AlibabaLLM(model_name=AlibabaModelName.qwen_turbo)
    if model_name == ModelType.local:
        llm = Local()
    langchain_llm = LangchainLLMWrapper(llm)
    langchain_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            'device': 'gpu' if is_available() else 'cpu',
        },
        encode_kwargs={
            'normalize_embeddings': True,
        }
    ))
    score = evaluate(dataset, llm=langchain_llm, embeddings=langchain_embeddings)
    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    score.to_pandas().to_csv(f'./evaluation/{time_stamp}_{model_name.value}_bge_WikiEval.csv')
    logger.info(f'评估完成')
except Exception as e:
    logger.error(e, exc_info=True)

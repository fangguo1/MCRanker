from pyserini.search import LuceneSearcher, get_topics, get_qrels
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from data_generator import data_generator
import tempfile
import os
import json
import shutil
THE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'trec-covid': 'beir-v1.0.0-trec-covid.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'webis-touche2020': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'nfcorpus': 'beir-v1.0.0-nfcorpus.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'dbpedia-entity': 'beir-v1.0.0-dbpedia-entity.flat',
    'fever': 'beir-v1.0.0-fever-flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'scifact': 'beir-v1.0.0-signal1m.flat',
    'trec-news': 'beir-v1.0.0-trec-news.flat',

    'mrtydi-ar': 'mrtydi-v1.1-arabic',
    'mrtydi-bn': 'mrtydi-v1.1-bengali',
    'mrtydi-fi': 'mrtydi-v1.1-finnish',
    'mrtydi-id': 'mrtydi-v1.1-indonesian',
    'mrtydi-ja': 'mrtydi-v1.1-japanese',
    'mrtydi-ko': 'mrtydi-v1.1-korean',
    'mrtydi-ru': 'mrtydi-v1.1-russian',
    'mrtydi-sw': 'mrtydi-v1.1-swahili',
    'mrtydi-te': 'mrtydi-v1.1-telugu',
    'mrtydi-th': 'mrtydi-v1.1-thai',
}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'trec-covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'webis-touche2020': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfcorpus': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia-entity': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'scifact': 'beir-v1.0.0-signal1m-test',

    'mrtydi-ar': 'mrtydi-v1.1-arabic-test',
    'mrtydi-bn': 'mrtydi-v1.1-bengali-test',
    'mrtydi-fi': 'mrtydi-v1.1-finnish-test',
    'mrtydi-id': 'mrtydi-v1.1-indonesian-test',
    'mrtydi-ja': 'mrtydi-v1.1-japanese-test',
    'mrtydi-ko': 'mrtydi-v1.1-korean-test',
    'mrtydi-ru': 'mrtydi-v1.1-russian-test',
    'mrtydi-sw': 'mrtydi-v1.1-swahili-test',
    'mrtydi-te': 'mrtydi-v1.1-telugu-test',
    'mrtydi-th': 'mrtydi-v1.1-thai-test',

}

def convert2trec(query,passages,scores):
    rank_list=[]
    for i in range(len(passages)):
        try:
            query[i][1]=int(query[i][1])
        except:
            pass

        rank_list.append({'query':query[i][0],
                          'hits':[]})
        for j in range(len(passages[i])):
            rank_list[i]['hits'].append({'content':f'',
                                 'qid':query[i][1],
                                 'docid':passages[i][j],
                                 'rank':j+1,
                                 'score':scores[query[i][1]][j]})


    return rank_list


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True

def run_retriever(topics, searcher, qrels=None, k=100):
    scores=defaultdict(list)
    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            hits = searcher.search(query, k=k)
            for hit in hits:
                scores[qid].append(hit.score)

    return scores

def get_scores(dataset):
    searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[dataset])
    topics = get_topics(THE_TOPICS[dataset] if dataset != 'dl20' else 'dl20')
    qrels = get_qrels(THE_TOPICS[dataset])
    order_topics = {}

    for key, value in qrels.items():
        order_topics[key] = topics[key]

    scores = run_retriever(order_topics, searcher, qrels, k=100)
    return scores

def get_ndcg(dataset,rank_list):
    from trec_eval import EvalFunction

    # Create an empty text file to write results, and pass the name to eval
    output_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(rank_list, output_file)
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[dataset], output_file])
    # Rename the output file to a better name
    shutil.move(output_file, f'eval_{dataset}.txt')

def get_ndcg_result(dataset,rank_list,output_file):
    from trec_eval import EvalFunction

    # Create an empty text file to write results, and pass the name to eval
    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(rank_list, temp_file )
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[dataset], temp_file])
    # Rename the output file to a better name
    shutil.move(temp_file , output_file)

# if __name__ =='__main__':
#     datasets = [('trec-covid', list(range(1,51))),
#                 ('webis-touche2020', list(range(1,50))),
#                 ('nfcorpus', list(range(1,324))),
#                 ('dbpedia-entity', list(range(1,401))),
#                 ('scifact', list(range(1,301))),
#                 #('scidocs', list(range(1,1001)))
#                 ]
#
#     for dataset in datasets:
#         scores = get_scores(dataset[0])
#         quer_qids = []
#         trec_docids = []
#         for i in dataset[1]:
#             dataset_path = './datasets/' + dataset[0]
#             query_file=f'query_{i}.txt'
#             pasasge_file=f'passages_bm25_100_{i}.txt'
#
#             dg=data_generator(dataset_path,verbose=False)
#             dg.load_query(os.path.join('orig',query_file))
#             dg.load_passages(os.path.join('orig',pasasge_file),dataset[0])
#
#             quer_qids.append([dg.query, dg.trec_qid])
#             trec_docids.append(dg.trec_docids)
#
#         rank_list=convert2trec(quer_qids,trec_docids,scores)
#         get_ndcg(dataset[0],rank_list)


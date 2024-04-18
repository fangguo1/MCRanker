import os
from baseline_helper import generate_message_json,run_multithread_gpt,load_gpt_pred_info
from prompt_helper import *
from prompt_builder import prompt_builder
from collections import defaultdict as ddict
from data_generator import data_generator
from compute_ndcg import get_scores, convert2trec, get_ndcg


def get_gpt_score(i,query,docid_doc_list,dataset_path):
    qbs_prompt = prompt_builder('score_0to10_no_reason_USE', load_from_warehouse=True,
                                warehouse_file='../prompt_warehouse.json')
    input_variables = ["passage", "question"]
    all_prompts = []
    pred_infos = []

    print(f"{i} :{query}")

    for (idx, doc) in docid_doc_list:
        input_values = [doc, query]
        input_dict = {}
        for key, value in zip(input_variables, input_values):
            input_dict[key] = value
        qbs_prompt.example_maker(input_variables=input_variables, input_dict=input_dict)
        all_prompts.append(qbs_prompt.prompt)
        pred_infos.append(idx)

    make_qbs_prompts(pred_infos, all_prompts, f'Multi_agent_{str(i)}_score_0to10_without_reason_gpt4_1106.json',
                     dataset_path, iterations=1)
    request_file = os.path.join(os.path.join(dataset_path, 'requests_bm25_100'),
                                f'Multi_agent_{str(i)}_score_0to10_without_reason_gpt4_1106.json')
    result_file = os.path.join(os.path.join(dataset_path, 'output_bm25_100'),
                               f'Multi_agent_{str(i)}_score_0to10_without_reason_gpt4_1106.json')
    run_multithread_gpt(request_file, result_file,api_key=API_KEY)


def make_qbs_prompts(pred_infos,all_prompts,file_name,dataset_path,iterations=1):
    output_request_json=os.path.join(os.path.join(dataset_path,'requests_bm25_100'),file_name)
    generate_message_json(pred_infos, all_prompts,output_request_path=output_request_json,iterations=iterations,model="gpt-4-1106-preview")
    return

def make_whole_ranking_list(reranker, dg, docid_results, doc_results,verbose=False):
    final_ranking = reranker.ranking(dg.query, zip(docid_results, doc_results))

    if verbose:
        for docid, doc in zip(docid_results, doc_results):
            print(docid, dg.docid2label[docid])

            print('Output:\n' + str(doc))

    return final_ranking



model_name = 'monot5'
reranker = Ranker(model_name)
reranker.load_ranker()

datasets=[('trec-covid',list(range(1,51))),
      ('webis-touche2020',list(range(1,50))),
      ('nfcorpus',list(range(1,324))),
      ('dbpedia-entity',list(range(1,401))),
      ('scifact',list(range(1,301))),
      ('signal',list(range(1,98))),
      ('news',list(range(1,58))),
      ('robust04',list(range(1,250)))
      ]

for dataset in datasets:
    scores = get_scores(dataset[0])
    quer_qid_top=[]
    trec_docids = []
    print(dataset[0])
    for i in dataset[1]:
        quer_qids = []

        query_file = f'query_{i}.txt'
        pasasge_file = f'passages_bm25_100_{i}.txt'
        dataset_path = f'../datasets/{dataset[0]}'
        dg = data_generator(dataset_path, verbose=False)
        dg.load_query(os.path.join('orig', query_file))
        dg.load_passages(os.path.join('orig', pasasge_file), dataset_path)
        print(i)
        print(dg.query)






        ### First Step: Generate Scores ###
        final_ranking = dg.docid2doc.keys()


        to_score_docs = []
        for j in range(len(final_ranking)):
            to_score_docs.append(final_ranking[j])

        print(f'Length of the passage set:{len(to_score_docs)}')


        docid_doc_list = []
        for docid in to_score_docs:
            docid_doc_list.append((docid, dg.docid2doc[docid]))

        get_gpt_score(i,dg.query,docid_doc_list,dataset_path)






        ### Second Step: Evaluation ###
        score_name=['gpt_0to10']
        scores_list = ddict(dict)

        for j in range(len(score_name)):
            result_file = os.path.join(os.path.join(dataset_path, 'output_bm25_100'),
                                       f'Multi_agent_{str(i)}_score_0to10_without_reason_gpt4_1106.json')
            preds = load_gpt_pred_info(result_file)

            for pred in preds:
                pred = [pred[0], pred[1]]
                docid = pred[0]
                try:
                    pred[1] = pred[1].replace('```', '')
                    if pred[1][:4] == 'json':
                        pred[1] = pred[1][4:]
                    ans = json.loads(pred[1])
                except Exception as e:
                    print(e)
                    print(f'Wrong with passage {docid} in json.loads!')
                    print(pred[1])
                    continue

                scores_list[score_name[j]][pred[0]] = ans['Score']

        final_scores=[]
        for j in range(30):
            key=list(scores_list['gpt_0to10'].keys())[j]
            final_scores.append((key,int(scores_list['gpt_0to10'][key])))

        final_scores.sort(key=lambda x: x[1],reverse=True)

        final_ranking_list=[]
        for j in range(len(final_scores)):
            final_ranking_list.append(final_scores[j][0])

        quer_qids.append([dg.query, dg.trec_qid])
        quer_qid_top.append([dg.query, dg.trec_qid])
        tmp_trec_docids = []

        for j in range(len(final_ranking_list)):
            tmp_trec_docids.append(dg.trec_docids[final_ranking_list[j]])
        trec_docids.append(tmp_trec_docids)



    rank_list = convert2trec(quer_qid_top, trec_docids, scores)

    get_ndcg(dataset[0], rank_list)

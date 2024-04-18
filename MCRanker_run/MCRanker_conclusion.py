import os
from openai import OpenAI
from tqdm import trange
from baseline_helper import generate_message_json,run_multithread_gpt,load_gpt_pred_info
from prompt_helper import *
from prompt_builder import prompt_builder
from collections import defaultdict as ddict
from data_generator import data_generator
from compute_ndcg import get_scores, convert2trec, get_ndcg

API_KEY='Your key'
client = OpenAI(
    api_key=API_KEY
)

def conclusion_agent(query, docid2doc, expert1_scores, expert2_scores, expert3_scores, score_name):
    qbs_prompt = prompt_builder('conclusion_0to10_3_agents_USE', load_from_warehouse=True,
                                warehouse_file='../prompt_warehouse.json')
    input_variables = [ "question","passage","Assessment of the three experts"]
    all_prompts = []
    pred_infos = []

    print(f"{i} :{query}")

    for j in range(len(expert1_scores)):
        score1=expert1_scores[j][1]
        score2=expert2_scores[j][1]
        score3=expert3_scores[j][1]
        doc=docid2doc[expert1_scores[j][0]]
        experts_assessment = f"\nAssessments from expert 1 who is an expert in {score_name[0]} : \nRelevance Score 1 (From 0 to 10):<<<{score1}>>> \n\nAssessments from expert 2 who is an expert in {score_name[1]} : \nRelevance Score 2 (From 0 to 10):<<<{score2}>>>\n\nAssessments from expert 3 who is an expert in {score_name[2]} : \nRelevance Score 3 (From 0 to 10):<<<{score3}>>>"
        input_values = [query,doc,experts_assessment]
        input_dict = {}
        for key, value in zip(input_variables, input_values):
            input_dict[key] = value
        qbs_prompt.example_maker(input_variables=input_variables, input_dict=input_dict)
        all_prompts.append(qbs_prompt.prompt)
        pred_infos.append(expert1_scores[j][0])

    make_qbs_prompts(pred_infos, all_prompts, f'Multi_agent_requests_conclusion_0to10_{str(i)}_gpt4_1106.json',
                     dataset_path, iterations=1)
    request_file = os.path.join(os.path.join(dataset_path, 'requests_bm25_100'),
                                f'Multi_agent_requests_conclusion_0to10_{str(i)}_gpt4_1106.json')
    result_file = os.path.join(os.path.join(dataset_path, 'output_bm25_100'),
                               f'Multi_agent_results_conclusion_0to10_{str(i)}_gpt4_1106.json')
    run_multithread_gpt(request_file, result_file,api_key=API_KEY)


def make_qbs_prompts(pred_infos,all_prompts,file_name,dataset_path,iterations=1):
    output_request_json=os.path.join(os.path.join(dataset_path,'requests_bm25_100'),file_name)
    generate_message_json(pred_infos, all_prompts,output_request_path=output_request_json,iterations=iterations,model="gpt-4-1106-preview")
    return




datasets=[('trec-covid',list(range(1,51))),
      ('webis-touche2020',list(range(1,50))),
      ('nfcorpus',list(range(1,324))),
      ('dbpedia-entity',list(range(1,401))),
      ('scifact',list(range(1,301))),
      ('signal',list(range(1,98))),
      ('news',list(range(1,58))),
      ('robust04',list(range(1,250)))
      ]

identity_num=2
for dataset in datasets:
    scores = get_scores(dataset[0])
    quer_qid_top=[]
    trec_docids = []
    print(dataset[0])
    for i in dataset[1]:
        print(i)
        quer_qids = []
        query_file = f'query_{i}.txt'
        pasasge_file = f'passages_bm25_100_{i}.txt'
        dataset_path = f'../datasets/{dataset[0]}'
        dg = data_generator(dataset_path, verbose=False)
        dg.load_query(os.path.join('orig', query_file))
        dg.load_passages(os.path.join('orig', pasasge_file), dataset_path)
        print(i)
        print(dg.query)






        ###First Step: Generate Scores ###
        with open(f'{dataset_path}/output_bm25_100/Multi_agent_{i}_identities_{str(identity_num)}_without_reason_gpt4_1106.json') as IN:
            identity_info_list = []
            identities = []
            for line in IN:
                line = line.strip()
                info = json.loads(line)
                identity_info_list.append(info)
                identities.append(info['name'])


        score_name = identities
        score_name.append('linguist')
        scores_list = ddict(dict)

        for j in range(len(score_name)):

            result_file = os.path.join(os.path.join(dataset_path, 'output_bm25_100'),
                                       f'Multi_agent_{str(i)}_results_{score_name[j]}_score_without_reason_gpt4_1106.json')
            preds = load_gpt_pred_info(result_file)

            for pred in preds:
                pred = [pred[0], pred[1]]
                docid = pred[0]
                try:
                    pred[1] = pred[1].replace('```', '')
                    pred[1] = pred[1].replace('."]', '."}')
                    if pred[1][:4] == 'json':
                        pred[1] = pred[1][4:]
                    ans = json.loads(pred[1])
                except Exception as e:
                    print(e)
                    print(f'Wrong with passage {docid} in json.loads!')
                    print(result_file)
                    print(pred[1])
                    continue

                scores_list[score_name[j]][pred[0]] = ans['Score']

        expert1_scores, expert2_scores, expert3_scores=[],[],[]
        for j in trange(30):
            key = list(scores_list[score_name[0]].keys())[j]
            expert1_scores.append((key,int(scores_list[score_name[0]][key])))
            expert2_scores.append((key, int(scores_list[score_name[1]][key])))
            expert3_scores.append((key, int(scores_list[score_name[2]][key])))


        conclusion_agent(dg.query, dg.docid2doc, expert1_scores, expert2_scores, expert3_scores, score_name)






        ###Second Step: Evaluation ###
        scores_list = ddict(dict)

        result_file = os.path.join(os.path.join(dataset_path, 'output_bm25_100'),
                                   f'Multi_agent_results_conclusion_0to10_{str(i)}_gpt4_1106.json')
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

            scores_list['conclusion'][pred[0]] = ans['Final_score']

        final_scores=[]
        if i in [45]:
            l=28
        else:
            l=30
        for j in range(l):
            key=list(scores_list['conclusion'].keys())[j]
            final_scores.append((key,int(scores_list['conclusion'][key])))

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


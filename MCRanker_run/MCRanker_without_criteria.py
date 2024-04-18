import os
from time import sleep
from openai import OpenAI
from baseline_helper import generate_message_json, run_multithread_gpt, load_gpt_pred_info
from prompt_helper import *
from prompt_builder import prompt_builder
from collections import defaultdict as ddict
from data_generator import data_generator
from compute_ndcg import get_scores, convert2trec, get_ndcg

API_KEY='Your key'
client = OpenAI(
    api_key=API_KEY
)

def use_gpt(model_name, input, key_list):
    messages = [{'role': 'user',
                 'content': input}]

    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name)
            print(f'{chat_completion.usage.prompt_tokens} prompt tokens counted by the OpenAI API.')
            bot_text = chat_completion.choices[0].message.content.strip()
            if key_list != []:
                bot_text = bot_text.replace('```', '')

                if bot_text[:4] == 'json':
                    bot_text = bot_text[4:]
                bot_text = json.loads(bot_text)
                for key in key_list:
                    assert key in bot_text
            break

        except Exception as e:
            print(e)
            sleep(0.1)
            continue

    return bot_text


def get_identity_score(i, query, docid_doc_list, dataset_path, identity):
    qbs_prompt = prompt_builder('identity_score_no_criteria_USE', load_from_warehouse=True,
                                warehouse_file='../prompt_warehouse.json')
    qbs_prompt.prefix = qbs_prompt.prefix.replace('#IDENTITY#', identity)

    input_variables = ["query",  "passage"]
    all_prompts = []
    pred_infos = []

    print(f"{i} :{query}")
    for (idx, doc) in docid_doc_list:
        input_values = [query, doc]
        input_dict = {}
        for key, value in zip(input_variables, input_values):
            input_dict[key] = value
        qbs_prompt.example_maker(input_variables=input_variables, input_dict=input_dict)
        all_prompts.append(qbs_prompt.prompt)
        pred_infos.append(idx)

    make_qbs_prompts(pred_infos, all_prompts,
                     f'Multi_agent_{str(i)}_requests_{identity}_score_without_criteria_gpt4_1106.json',
                     dataset_path, iterations=1)
    request_file = os.path.join(os.path.join(dataset_path, 'requests_bm25_100'),
                                f'Multi_agent_{str(i)}_requests_{identity}_score_without_criteria_gpt4_1106.json')
    result_file = os.path.join(os.path.join(dataset_path, 'output_bm25_100'),
                               f'Multi_agent_{str(i)}_results_{identity}_score_without_criteria_gpt4_1106.json')
    run_multithread_gpt(request_file, result_file,api_key=API_KEY)



def get_linguist_score(i,query,docid_doc_list,dataset_path):
    qbs_prompt = prompt_builder('linguist_score_no_criteria_USE', load_from_warehouse=True,
                                warehouse_file='../prompt_warehouse.json')
    input_variables = [ "question","passage"]
    all_prompts = []
    pred_infos = []

    print(f"{i} :{query}")

    for (idx, doc) in docid_doc_list:
        input_values = [query,doc]
        input_dict = {}
        for key, value in zip(input_variables, input_values):
            input_dict[key] = value
        qbs_prompt.example_maker(input_variables=input_variables, input_dict=input_dict)
        all_prompts.append(qbs_prompt.prompt)
        pred_infos.append(idx)

    make_qbs_prompts(pred_infos, all_prompts, f'Multi_agent_{str(i)}_requests_linguist_score_without_criteria_gpt4_1106.json',
                     dataset_path, iterations=1)
    request_file = os.path.join(os.path.join(dataset_path, 'requests_bm25_100'),
                                f'Multi_agent_{str(i)}_requests_linguist_score_without_criteria_gpt4_1106.json')
    result_file = os.path.join(os.path.join(dataset_path, 'output_bm25_100'),
                               f'Multi_agent_{str(i)}_results_linguist_score_without_criteria_gpt4_1106.json')
    run_multithread_gpt(request_file, result_file,api_key=API_KEY)

def generate_identities(query, passage, number, model_name='gpt-4-1106-preview'):
    instruction = f"The task is when given a query and a passage example, try to guess the most probable user identities. You can name up to {number} identities. Try to make these identities very different from each other. Note that the passage example is just to show you the passage style and the user could possiblely like/dislike it. So do not let the stance in the passage to influence your guess." \
                  f"\nGiven query:<<<{query}>>>" \
                  f"\nGiven passage:<<<{passage}>>>" \
                  "\nPlease give your output in JSON format with keys are \"Identities\" and \"Reason\" ." \
                  "\nUnder the content of \"Identities\",please output the identity,your identity should be correct,clear,and easy to understand " \
                  "\nUnder the content of \"Reason\",explain why you output these identities." \
                  "\nProvide a clear and concise response,just give your answer in JSON format as I request,don't say any other words."
    key_list = ['Identities', 'Reason']

    bot_text = use_gpt(model_name, instruction, key_list)

    return bot_text['Identities'], bot_text['Reason']


def generate_criteria_per_identity(identity, query, model_name='gpt-4-1106-preview'):
    instruction = f"As a {identity},you are ask to judge relevance of a passage in relation to the given query, list the criteria for judging relevance between the query and a passage, assigning weights to each criterion. " \
                  f"\nGiven query:<<<{query}>>>" \
                  "\nPlease give your output in JSON format with keys are \"Criteria\" and \"Reason\" ." \
                  f"\nUnder the content of \"Criteria\",please output the criteria and some explanation to it,your criteria should be correct,clear,executable and easy to understand,the criteria should come from your knowledge as the {identity}.The weight to every criterion should be added after each criterion with 'The weight to this criterion is:'." \
                  "\nUnder the content of \"Reason\",explain why you output these criteria." \
                  f"\nFocus solely on your identity and avoid deductions outside of your {identity}. " \
                  "\nProvide a clear and concise response,just give your answer in JSON format as I request,don't say any other words."

    key_list = ['Criteria', 'Reason']

    bot_text = use_gpt(model_name, instruction, key_list)

    return bot_text['Criteria'], bot_text['Reason']


def make_qbs_prompts(pred_infos, all_prompts, file_name, dataset_path, iterations=1):
    output_request_json = os.path.join(os.path.join(dataset_path, 'requests_bm25_100'),
                                       file_name)
    generate_message_json(pred_infos, all_prompts, output_request_path=output_request_json, iterations=iterations,
                          model='gpt-4-1106-preview')
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

identity_num = 2

for dataset in datasets:
    scores = get_scores(dataset[0])
    quer_qid_top = []
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






        ### First Step: Generate Scores ###
        final_ranking = dg.docid2doc.keys()


        to_score_docs = []
        for j in range(len(final_ranking)):
            to_score_docs.append(final_ranking[j])

        print(f'Length of the passage set:{len(to_score_docs)}')


        to_score_docs = []
        for j in range(len(final_ranking[:30])):
            to_score_docs.append(final_ranking[j])

        print(f'Length of the passage set:{len(to_score_docs)}')


        ### GENERATE IDENTITIES ###
        identities_raw,reason=generate_identities(dg.query,dg.docid2doc[2],identity_num,'gpt-4-1106-preview')
        identities=[]
        for iden in identities_raw:
            iden=iden.lower().replace(' ','_')
            if '/' in iden:
                iden=iden.split('/')[1]
            identities.append(iden)

        with open(f'{dataset_path}/output_bm25_100/Multi_agent_{i}_identities_without_criteria_gpt_1106_{str(identity_num)}.txt','w+') as OUT:
            for iden in identities:
                iden=iden.replace(' ','_')
                iden=iden.lower()
                OUT.write(iden+'\n')

        ### GENERATE CRITERIA ###
        identity_info_list=[]
        for identity in identities:
            criteria,reason=generate_criteria_per_identity(identity,dg.query,model_name='gpt-4-1106-preview')
            identity_info={'name':identity,'criteria':criteria,'reason':reason}
            identity_info_list.append(identity_info)
        with open(f'{dataset_path}/output_bm25_100/Multi_agent_{i}_identities_{str(identity_num)}_without_criteria_gpt_1106.json','w+') as f:
            for identity_info in identity_info_list:
                json_string = json.dumps(identity_info)
                f.write(json_string + "\n")


        with open(f'{dataset_path}/output_bm25_100/Multi_agent_{i}_identities_{str(identity_num)}_without_criteria_gpt_1106.json') as IN:
            identity_info_list=[]
            identities=[]
            for line in IN:
                line=line.strip()
                info=json.loads(line)
                identity_info_list.append(info)
                identities.append(info['name'])

        ### GENERATE SCORE FOR EACH IDENTITY ###
        docid_doc_list = []
        for docid in to_score_docs:
            docid_doc_list.append((docid, dg.docid2doc[docid]))

        get_linguist_score(i,dg.query, docid_doc_list, dataset_path)

        for tp in identity_info_list:
            identity=tp['name']
            get_identity_score(i,dg.query,docid_doc_list,dataset_path,identity)






        ### Second Step: Evaluation ###
        with open(f'{dataset_path}/output_bm25_100/Multi_agent_{i}_identities_{str(identity_num)}_without_criteria_gpt_1106.json') as IN:
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
            print(score_name[j])
            result_file = os.path.join(os.path.join(dataset_path, 'output_bm25_100'),
                                       f'Multi_agent_{str(i)}_results_{score_name[j]}_score_without_criteria_gpt4_1106.json')
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

        final_scores = []
        selected_identities = score_name
        rank_ensembles = ddict(float)
        all_score_tps = []

        for j in range(30):
            key = list(scores_list[score_name[0]].keys())[j]
            identity_scores = []
            final_score = 0
            for identity, score_dict in scores_list.items():
                if identity in selected_identities:
                    try:
                        final_score += score_dict[key]
                    except:
                        final_score += 5
                        print(identity, key)
                        continue
            final_scores.append((key, final_score))

        final_scores.sort(key=lambda x: x[1], reverse=True)

        final_ranking_list = []
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


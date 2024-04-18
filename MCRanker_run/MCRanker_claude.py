import json
from time import sleep
import anthropic
from data_generator import data_generator
from ranker import Ranker
import os
import asyncio
from anthropic import AsyncAnthropic

API_KEY='Your key'

client_claude_async = AsyncAnthropic(
    api_key=API_KEY,
)

client_claude = anthropic.Anthropic(
    api_key=API_KEY,
)


def use_claude(model_name, input, key_list):
    messages = [{'role': 'user',
                 'content': input}]
    while True:
        try:
            message = client_claude.messages.create(
                model=model_name,
                max_tokens=500,
                temperature=0.0,
                messages=messages
            )

            bot_text = message.content[0].text.strip()
            if 'I cannot provide' in bot_text or "I can't provide" in bot_text:
                bot_text = 'None'
                break
            bot_text=bot_text.replace('\n',' ')
            if key_list != []:
                bot_text = json.loads(bot_text)
                for key in key_list:
                    assert key in bot_text
            break

        except Exception as e:
            print(e)
            sleep(1)
            continue

    return bot_text


def linguist_criteria(query,model_name):
    instruction ="As an expert in linguist,you are good at judging linguist relevance of a passage in relation to the given query, list the criteria for judging linguist relevance between the query and a passage, assigning weights to each criterion.  " \
                f"\nGiven query:<<<{query}>>>" \
                 "\nPlease give your output in JSON format with keys are \"Criteria\" and \"Reason\" ." \
                 "\nUnder the content of \"Criteria\",please output the criteria and some explanation to it,your criteria should be correct,clear,executable and easy to understand,the criteria should come from your knowledge as the linguist expert.The weight to every criterion should be added after each criterion with 'The weight to this criterion is:'." \
                 "\nUnder the content of \"Reason\",explain why you output these criteria." \
                 "\nFocus solely on linguist expert and avoid deductions outside of your expert criteria. " \
                 "\nProvide a clear and concise response,just give your answer in JSON format as I request,don't say any other words."

    key_list=['Criteria','Reason']

    bot_text=use_claude(model_name,instruction,key_list)
    if bot_text == 'None':
        return bot_text

    return [bot_text['Criteria'],bot_text['Reason']]


def generate_identities(query, passage, number, model_name):
    instruction = f"The task is when given a query and a passage example, try to guess the most probable user identities. You can name up to {number} identities. Try to make these identities very different from each other. Note that the passage example is just to show you the passage style and the user could possiblely like/dislike it. So do not let the stance in the passage to influence your guess." \
                  f"\nGiven query:<<<{query}>>>" \
                  f"\nGiven passage:<<<{passage}>>>" \
                  "\nPlease give your output in JSON format with keys are \"Identities\" and \"Reason\" ." \
                  f"\nUnder the content of \"Identities\",please output the {number} identities in a python list,your identity should be correct,clear,and easy to understand " \
                  "\nUnder the content of \"Reason\",explain why you output these identities." \
                  "\nProvide a clear and concise response,just give your answer in JSON format as I request,don't say any other words."
    key_list = ['Identities', 'Reason']

    bot_text = use_claude(model_name, instruction, key_list)
    if bot_text == 'None':
        return bot_text

    return [bot_text['Identities'], bot_text['Reason']]


def generate_criteria_per_identity(identity, query, model_name):
    instruction = f"As a {identity},you are ask to judge relevance of a passage in relation to the given query, list the criteria for judging relevance between the query and a passage, assigning weights to each criterion. " \
                  f"\nGiven query:<<<{query}>>>" \
                  "\nPlease give your output in JSON format with keys are \"Criteria\" and \"Reason\" ." \
                  f"\nUnder the content of \"Criteria\",please output the criteria and some explanation to it,your criteria should be correct,clear,executable and easy to understand,the criteria should come from your knowledge as the {identity}.The weight to every criterion should be added after each criterion with 'The weight to this criterion is:'." \
                  "\nUnder the content of \"Reason\",explain why you output these criteria." \
                  f"\nFocus solely on your identity and avoid deductions outside of your {identity}. " \
                  "\nProvide a clear and concise response,just give your answer in JSON format as I request,don't say any other words."

    key_list = ['Criteria', 'Reason']

    bot_text = use_claude(model_name, instruction, key_list)
    if bot_text == 'None':
        return bot_text

    return [bot_text['Criteria'], bot_text['Reason']]

def make_whole_ranking_list(reranker, dg, docid_results, doc_results, verbose=False):
    final_ranking = reranker.ranking(dg.query, zip(docid_results, doc_results))

    if verbose:
        for docid, doc in zip(docid_results, doc_results):
            print(docid, dg.docid2label[docid])
            print('Output:\n' + str(doc))

    return final_ranking

async def send_command(id,command):
    while True:
        try:
            message = await client_claude_async.messages.create(
                max_tokens=50,
                messages=command,
                model="claude-3-sonnet-20240229",
            )
            bot_text=message.content[0].text

            break
        except Exception as e:
            print(e)
            continue

    return [id,bot_text]



async def get_identity_score(i,query,docid_doc_list,dataset_path,identity,criteria):
    commands=[]
    for j in range(len(docid_doc_list)):
        prompt=[{"role": "user", "content": f"You are a helpful assistant.\n        Ignore all previous instructions.\n        \n        Role:You are a {identity}\n        \n        Task Desciption:I will give you the criteria, one passage and a query, you should follow the criteria to analyse the given passage's relevance from the view as a curious_individual. Generate your answer in json format,with the key is \"Score\" . Under the content of \"Score\" ,give me an integer score from 0 to 10 to represent the linguist expert's view relevance degree, just give the score, don't say any other words .\n        \n        Input-Output Description:The input will be the criteria,a passage and a query,generate your output as I request:\n        \n        \n\n        Your Answer:\n\n\n        \n        query: {query}\n        \n        criteria: {criteria}\n        \n        passage: {docid_doc_list[j][1]}\n        \n        Output: \n        "}]
        commands.append([docid_doc_list[j][0],prompt])
    results = await asyncio.gather(*(send_command(command[0],command[1]) for command in commands))

    with open(f"{dataset_path}/output_bm25_100/Multi_agent_{str(i)}_{identity}_score_without_reason_claude.json",'w+') as f:
        for j in range(len(results)):
            content={"pred_info": results[j][0], "prediction": results[j][1]}
            json_content=json.dumps(content)
            f.write(json_content+'\n')


async def get_linguist_score(i,query, docid_doc_list, dataset_path,criteria):
    commands=[]
    for j in range(len(docid_doc_list)):
        prompt=[ {"role": "user", "content": f"You are a helpful assistant.\n        Ignore all previous instructions.\n        \n        Role:You are an expert in linguist.\n        \n        Task Desciption:I will give you the criteria you should obey,one passage and a query, you should obey the criteria,try to analyse the given passage's relevance from the view of an linguist expert. Generate your answer in json format,with the key is \"Score\". Under the content of \"Score\" ,give me an integer score from 0 to 10 to represent the linguist expert's view relevance degree, just give the score, don't say any other words .\n        \n        Input-Output Description:The input will be the criteria,a passage and a query,generate your output as I request:\n        \n        \n\n        Your Answer:\n\n\n        \n        criteria: {criteria}\n        \n        passage: {docid_doc_list[j][1]}\n        \n        question: {query}\n        \n        Output: \n        "}]
        commands.append([docid_doc_list[j][0],prompt])
    results = await asyncio.gather(*(send_command(command[0],command[1]) for command in commands))

    with open(f"{dataset_path}/output_bm25_100/Multi_agent_{str(i)}_linguist_score_without_reason_claude.json",'w+') as f:
        for j in range(len(results)):
            content={"pred_info": results[j][0], "prediction": results[j][1]}
            json_content=json.dumps(content)
            f.write(json_content+'\n')

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

identity_num = 2

for dataset in datasets:
    print(dataset[0])
    for i in dataset[1]:


        query_file = f'query_{i}.txt'
        pasasge_file = f'passages_bm25_100_{i}.txt'
        dataset_path = f'../datasets/{dataset[0]}'
        dg = data_generator(dataset_path, verbose=False)
        dg.load_query(os.path.join('orig', query_file))
        dg.load_passages(os.path.join('orig', pasasge_file), dataset_path)
        print(dg.query)
        print(i)
        print(dg.query)






        ### First Step: Generate Scores ###
        final_ranking = dg.docid2doc.keys()


        to_score_docs = []
        for j in range(len(final_ranking)):
            to_score_docs.append(final_ranking[j])

        print(f'Length of the passage set:{len(to_score_docs)}')


        tmp=linguist_criteria(dg.query,"claude-3-sonnet-20240229")
        if tmp == 'None':
            lin_criteria=tmp
        else:
            lin_criteria=tmp[0]

        tmp=generate_identities(dg.query,dg.docid2doc[2],identity_num,"claude-3-sonnet-20240229")
        if tmp == 'None':
            identities_raw = [tmp]
        else:
            identities_raw = tmp[0]
        identities=[]
        for iden in identities_raw:
            iden=iden.lower().replace(' ','_')
            if '/' in iden:
                iden=iden.split('/')[1]
            identities.append(iden)

        with open(f'{dataset_path}/output_bm25_100/Multi_agent_{i}_identities_without_reason_{str(identity_num)}_claude.txt','w+') as OUT:
            for iden in identities:
                iden=iden.replace(' ','_')
                iden=iden.lower()
                OUT.write(iden+'\n')


        identity_info_list=[]
        for identity in identities:
            tmp=generate_criteria_per_identity(identity,dg.query,model_name="claude-3-sonnet-20240229")
            if tmp == 'None':
                criteria, reason =tmp,tmp
            else:
                criteria, reason =tmp[0],tmp[1]

            identity_info={'name':identity,'criteria':criteria,'reason':reason}
            identity_info_list.append(identity_info)

        with open(f'{dataset_path}/output_bm25_100/Multi_agent_{i}_identities_{str(identity_num)}_without_reason_claude.json','w+') as f:
            for identity_info in identity_info_list:
                json_string = json.dumps(identity_info)
                f.write(json_string + "\n")


        with open(f'{dataset_path}/output_bm25_100/Multi_agent_{i}_identities_{str(identity_num)}_without_reason_claude.json') as IN:
            identity_info_list=[]
            identities=[]
            for line in IN:
                line=line.strip()
                info=json.loads(line)
                identity_info_list.append(info)
                identities.append(info['name'])


        docid_doc_list = []
        for docid in to_score_docs:
            docid_doc_list.append((docid, dg.docid2doc[docid]))

        asyncio.run(get_linguist_score(i,dg.query, docid_doc_list, dataset_path,str(lin_criteria)))

        for tp in identity_info_list:
            identity=tp['name']
            criteria=tp['criteria']
            asyncio.run(get_identity_score(i,dg.query,docid_doc_list,dataset_path,identity,criteria))






        ### Second Step: Evaluation ###
        # Evaluation are just as gpt series code.
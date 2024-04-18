import json
from collections import defaultdict as ddict

def append_to_prompt_warehouse_info(prompt_name,role,task,io,fs_variables,fs_examples,warehouse_file='prompt_warehouse.json'):
    temp_dict=ddict(str)
    temp_dict['prompt_name']=prompt_name
    temp_dict['role']=role
    temp_dict['task']=task
    temp_dict['io']=io
    temp_dict['fs_variables']=fs_variables
    temp_dict['fs_examples']=fs_examples
    with open(warehouse_file, "a") as f:
        json_string = json.dumps(temp_dict)
        f.write(json_string + "\n")

def load_from_prompt_warehouse_info(input_prompt_name,warehouse_file='prompt_warehouse.json'):
    preds=[]
    with open(warehouse_file) as IN:
        cnt=0
        for line in IN:
            #print(line)
            json_info=json.loads(line)
			#print(json_info)
            prompt_name=json_info['prompt_name']
            if prompt_name!=input_prompt_name:
                continue
            task=json_info['task']
            io=json_info['io']
            role=json_info['role']
            fs_variables=json_info['fs_variables']
            fs_examples=json_info['fs_examples']
    return prompt_name,task,io,role,fs_variables,fs_examples
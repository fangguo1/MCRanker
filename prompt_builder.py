from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from prompt_helper import *
def make_triple_quotes_from_list(l):
    temp=""
    for i in l:
        temp+=f"""
        {i}: {{{i}}}
        """
    return temp

class prompt_builder:
    def __init__(self,prompt_name,load_from_warehouse=False,warehouse_file=None):
        
        if load_from_warehouse:
            self.load_from_warehouse=True
            assert prompt_name!=None
            prompt_name,task,io,role,fs_variables,fs_examples=load_from_prompt_warehouse_info(input_prompt_name=prompt_name,warehouse_file=warehouse_file)
            self.name=prompt_name
            self.prefix_maker(role=role,task=task,io=io)
            self.fs_variables=fs_variables
            self.fs_examples=fs_examples
        else:
            self.load_from_warehouse=False
            self.prompt_name=prompt_name
		
    def prefix_maker(self,role,task,io):
    
        prefix_template = """
        Ignore all previous instructions.
        
        Role:{role}
        
        Task Desciption:{task}
        
        Input-Output Description:{io}
        
        """
        prompt_template = PromptTemplate(
            input_variables=["role","task","io"],
            template=prefix_template
        )
        prefix=prompt_template.format(role=role,task=task,io=io)
        self.prefix=prefix
    
    def example_maker(self,input_variables,input_dict,examples=None,example_variables=None):
        if self.load_from_warehouse:
            examples=self.fs_examples
            example_variables=self.fs_variables
        example_template=make_triple_quotes_from_list(example_variables)
        example_template=f"""
        
        {example_template}
        
        """
        if len(examples)==0:
            example_template=''
        example_prompt = PromptTemplate(
            #input_variables=["aspect", "list_of_phrases","k","output"],
            input_variables=example_variables,
            template=example_template
            )
        
        suffix = """
        Your Answer:\n\n
        """
        suffix+=make_triple_quotes_from_list(input_variables)
        suffix+="""
        Output: 
        """
        #suffix='"""\n'+suffix+'"""'
        # now create the few shot prompt template
        if len(examples)==0:
            prefix=self.prefix
        else:
            prefix=self.prefix+'I will give some examples for you to understand the output format.  Here is the example:'
        few_shot_prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            example_separator="\n"
        )
        prompt=few_shot_prompt_template.format(**input_dict)
        self.prompt=prompt
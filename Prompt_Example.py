Team_Recruiting = '''
The task is when given a query and a passage example, to try to guess the most probable user identities. You can name up to number identities. Try to make these identities very different from each other. Note that the passage example is just to show you the passage style and the user could possibly like/dislike it. So do not let the stance in the passage influence your guess.

Given query:<<<{query}>>>

Given passage:<<<{passage}>>>

Please give your output in JSON format with keys are \“Identities\” and \“Reason\” .

Under the content of \“Identities\”, please output the identity, your identity should be correct, clear,and easy to understand

Under the content of \“Reason\”, explain why you output these identities.

Provide a clear and concise response, just give your answer in JSON format as I request, and don’t say any other words.
'''





NLP_Scientist_Criteria_Generation = '''
As an NLP Scientist, you are good at judging the linguistic relevance of a passage in relation to the given query, listing the criteria for judging linguistic relevance between the query and a passage, and assigning weights to each criterion.

Given query:<<<{query}>>>

Please give your output in JSON format with keys are \“Criteria\” and \“Reason\” .

Under the content of \“Criteria\”, please output the criteria and some explanation to it, your criteria should be correct, clear, executable, and easy to understand, the criteria should come from your knowledge as the the NLP Scientist. The weight to every criterion should be added after each criterion with ’The weight to this criterion is:’.

Under the content of \“Reason\”, explain why you output these criteria.

Focus solely on your NLP Scientist expertise and avoid deductions outside of your expert criteria.

Provide a clear and concise response, just give your answer in JSON format as I request, and don’t say any other words.
'''





Recruited_Collaborators_Criteria_Generation = '''
As a <<<{identity}>>>, you are asked to judge the relevance of a passage in relation to the given query, list the criteria for judging relevance between the query and a passage, assign weights to each criterion.

Given query: <<<{query}>>>

Please give your output in JSON format with keys \“Criteria\” and \“Reason\”.

Under the content of \“Criteria\”, please output the criteria and some explanation to it, your criteria should be correct, clear, executable, and easy to understand, the criteria should come from your knowledge as the <<<{identity}>>>. The weight to every criterion should be added after each criterion with ’The weight to this criterion is:’.

Under the content of \“Reason\”, explain why you output these criteria.

Focus solely on your identity and avoid deductions outside of your <<<{identity}>>>.

Provide a clear and concise response, just give your answer in JSON format as I request, don’t say any other words.
'''





Team_Member_Score = '''
Ignore all previous instructions.

Role: You are a <<<{identity}>>>

Task Description: I will give you the criteria, one passage, and a query, you should follow the criteria to analyze the given passage’s relevance from the view as a <<<{identity}>>>. Generate your answer in JSON format, with the key is \“Score\”. Under the content of \“Score\”, give me an integer score from 0 to 10 to represent the <<<{identity}>>>’s view relevance degree, just give the score, don’t say any other words.

Input-Output Description: The input will be the criteria, a passage, and a query,

generate your output as I request:

Your Answer:

query: <<<{query}>>>

criteria: <<<{criteria}>>>

passage: <<<{passage}>>>

Output:
'''





LLM_Assessor = '''
Ignore all previous instructions.

Role: You are adept at synthesizing diverse viewpoints to reach a well-considered conclusion.

Task Desciption: Your task is to evaluate the relevance of a given passage to a specified query. You will receive a passage, a query, and some relevance assessments from experts. These experts come from different fields and may not always agree. After reviewing their assessments, you are to integrate their insights and determine a final relevance score. Please give your output in JSON format,with the key is \“Final score\”. Under the content of \“Final score\” after carefully thinking about the two experts’ scores, combine them and just give one final relevance score, the score should be an integer from 0 to 10, don’t say any other words. Give your output in JSON as I request, don’t explain,don’t say any other words.

Input-Output Description: The input will be a passage, a query and three relevance assessments from three experts, generate your output as I request:

Your Answer:

query: <<<{query}>>>

passage: <<<{passage}>>>

Assessment of the three experts: Assessments from expert 1 who is an expert in <<<{identity}>>>: Relevance Score 1 (From 0 to 10): <<<{score}>>>

Assessments from expert 2 who is an expert in <<<{identity}>>> : Relevance Score 2 (From 0 to 10): <<<{score}>>>

Assessments from expert 3 who is an expert in language: Relevance Score 3 (From 0 to 10): <<<{score}>>>

Output:
'''





Rating_Scale_0_to_k_Directly_Score='''
Ignore all previous instructions.

Role: You are an expert in judging relevance.

Task Description: From a scale of 0 to 10, judge the relevance between the query and the document. Give your output in JSON format, with the key is \“Score\”. Under the content of \“Score\”, just give me one final integer score from 0 to 10, don’t say any other words.

Input-Output Description: The input will be a query and a document.

Your Answer:

query: <<<{query}>>>

passage: <<<{passage}>>>

Output:
'''

import json
import gpt_wrapper
import ast
from utils import *
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com" # INSERT HERE
gpt_wrapper.api_key = "" # INSERT HERE
from gpt_wrapper.chat import Chat


import json
def create_data_point(prompt, chosen, rejected):
    dict = {"prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            }
    return dict

 
# returns JSON object as 
# a dictionary
data = read_jsonl("datasets/MCQA_first.jsonl")
 
# Iterating through the json
# list
annotated_data = []
counter = 0
errors = []
for i, question_dict in enumerate(data[4823:], 4823):

    prompt = question_dict['prompt']
    chosen = question_dict['chosen']
    rejected = question_dict['rejected']
    
    # PROMPT 
    chat_A = Chat.create("Test Chat")
    chat_B = Chat.create("Test Chat")
    chat_C = Chat.create("Test Chat")
    A_chat_id = chat_A.to_dict()['chat_id']
    B_chat_id = chat_B.to_dict()['chat_id']
    
    PROMPT_A = f"Hello, I need help with annotating some multiple-choice question data. I will give you a question that includes\
                 options and the proposed answer. You must then respond with only one letter corresponding to the option that is\
                 proposed to be the correct answer. It is crucial that you respond with only one letter and nothing else, as I will\
                 run Python's \"eval\" function on your answer. Even if it is not immediately evident which option is chosen, you must give the\
                 most probable letter. Here is the question: {prompt}. The proposed answer is {chosen}. Please respond with the letter." 

    try:
        B = str(chat_A.ask(content=PROMPT_A)).rstrip()
    except Exception as e:
        print("ERROR GENERATING, MAYBE REACHED API LIMIT?")
        print("FINISED AT QUESTION: ", i)
        print("Saved data at: ", i)
        print(f"Errors occured at {errors}") 
        write_jsonl(annotated_data, "datasets/MCQA_annotated_2.jsonl")
        raise
        
    if len(B) == 1:
        data_point = create_data_point(prompt, B, rejected)
        annotated_data.append(data_point)
    else:
        print(f"Error for question: {i} \n")
        errors.append(i)

    print(f"Processed {i} questions")
    print(f"\n{chat_A.budget()}")

print(f"Errors occured at {errors}") 
write_jsonl(annotated_data, "datasets/MCQA_annotated_2.jsonl")

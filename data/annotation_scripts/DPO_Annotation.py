import json
import gpt_wrapper
import ast

gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com" # INSERT HERE
gpt_wrapper.api_key = "" # INSERT HERE
from gpt_wrapper.chat import Chat


import json
def create_data_point(course_id, question_id, question, A_chat_id, B_chat_id, A, B, ranking_criteria):
    dict = {"course_id": course_id,
            "question_id": question_id,
            "question": question,
            "A_chat_id": A_chat_id,
            "B_chat_id": B_chat_id,
            "A": A,
            "B": B,
            "ranking_criteria": ranking_criteria
            }
    return dict

 
# Opening JSON file
f = open('') # INSERT HERE
 
# returns JSON object as 
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
annotated_data = []
counter = 0
errors = []
for question_dict in data:
    course_id = question_dict['course_id']
    question_id = question_dict['question_id']
    question = question_dict['question_body']
    options = question_dict["question_options"]
    
    # PROMPT 
    chat_A = Chat.create("Test Chat")
    chat_B = Chat.create("Test Chat")
    chat_C = Chat.create("Test Chat")
    A_chat_id = chat_A.to_dict()['chat_id']
    B_chat_id = chat_B.to_dict()['chat_id']
    if options is None:
        PROMPT_A_1 = f"Here is a technical question from a STEM course:\"{question}\" \
        Please analyze the question and identify the main subject area it pertains to within these fields of \
        science. Also, outline the key points or concepts that are necessary for solving this question."
                
        PROMPT_A_2 = f"Using the subject area and key points you identified from the earlier question, provide a \
                    detailed step-by-step solution to the problem: \"{question}\". Consider each step in \
                    the process, and explain how you arrive at the solution, including any necessary algorithms, \
                    calculations, or theoretical considerations. Take a deep breath and work on this problem step-by-step. \
                    Note that any math should be written in LaTeX."

        PROMPT_B = f"Here is a technical question from a STEM course:\"{question}\" provide a \
                    solution to the problem, including an explanation. Note that any math should be written in LaTeX."
                    
    else:
        PROMPT_A_1 = f"Here is a technical question from a STEM course:\"{question}\". This is a multiple choice question \
                with the following options: { {*options} }.Please analyze the question and identify the main subject \
                area it pertains to within these fields of science. Also, outline the key points or concepts that are \
                necessary for solving this question."
                
        PROMPT_A_2 = f"Using the subject area and key points you identified from the earlier question, provide a \
                    detailed step-by-step solution to the problem: \"{question}\". Consider each step in \
                    the process, and explain how you arrive at the solution, including any necessary algorithms, \
                    calculations, or theoretical considerations. This is a multiple choice question so answer clearly \
                    which alternative of { {*options} } is correct before giving your explanations. Take a deep breath \
                    and work on this problem step-by-step.Note that any math should be written in LaTeX."

        PROMPT_B = f"Here is a technical question from a STEM course:\"{question}\" provide a \
                    solution to the problem, including an explanation. This is a multiple choice question so answer clearly \
                    which alternative of { {*options} } is correct before giving your explanations. Note that any math \
                    should be written in LaTeX."
    
    
    A_1 = chat_A.ask(content=PROMPT_A_1)
    A_2 = str(chat_A.ask(content=PROMPT_A_2))
    
    B_1 = str(chat_B.ask(content=PROMPT_B))

    
    if options is None:
        PROMPT_C = f"I will provide you a question from a STEM course, and two possible solutions, \"A\" and \"B\". \
I want you to evaluate which solution is best in terms of the correctness, relevance, clarity, and completeness. \
Then you will rank the solutions overall, based on these criteria. You will answer this prompt precisely \
in this format \"{{\"overall\": \"A\", \"correctness\": \"B\", \"relevance\": \
\"AB\", \"clarity\": \"A\", \"completeness\": \"A\", \"other\": \"Conciseness: B; Engagement: AB\"}}\" \
This is the only thing you should write in your response, so that I can run pythons eval on you response. \
For example \"correctness\": \"B\" implies that \"B\" is prefered over \"A\" in terms of correctness, \
whereas \"relevance\": \"AB\" implies both \"A\" and \"B\" are relevant. The \"correctness\" is the most important criteria \
in order to determine \"overall\", but the other criterias are also important. In the rare case that none of the answers fulfill \
a certain criteria you could answer with \" None \", for example \"correctness\": \"None\". You are allowed to leave \"other\" \
empty. Here is the question: \"{question}\". Here are the two solutions. Solution A: \"{A_2}\" \n -----------------\n \
Solution B: \"{B_1}\".\n -----------------\nNow evaluate the solutions, and answer in the requested format."
                    
    else:
        PROMPT_C = f"I will provide you a question from a STEM course, and two possible solutions, \"A\" and \"B\". \
I want you to evaluate which solution is best in terms of the correctness, relevance, clarity, and completeness. \
Then you will rank the solutions overall, based on these criteria. You will answer this prompt precisely \
in this format \"{{\"overall\": \"A\", \"correctness\": \"B\", \"relevance\": \
\"AB\", \"clarity\": \"A\", \"completeness\": \"A\", \"other\": \"Conciseness: B; Engagement: AB\"}}\" \
This is the only thing you should write in your response, so that I can run pythons eval on you response. \
For example \"correctness\": \"B\" implies that \"B\" is prefered over \"A\" in terms of correctness, \
whereas \"relevance\": \"AB\" implies both \"A\" and \"B\" are relevant. The \"correctness\" is the most important criteria \
in order to determine \"overall\", but the other criterias are also important. In the rare case that none of the answers fulfill \
a certain criteria you could answer with \" None \", for example \"correctness\": \"None\". You are allowed to leave \"other\" \
empty. Here is the question: \"{question}\". This is a multiple choice question with the alternatives { {*options} } \
Here are the two solutions. Solution A: \"{A_2}\" \n -----------------\nSolution B: \"{B_1}\".\n -----------------\nNow \
evaluate the solutions, and answer in the requested format."
    
    try:
        C = ast.literal_eval(str(chat_C.ask(content=PROMPT_C)))       
        data_point = create_data_point(course_id, question_id, question, A_chat_id, B_chat_id, A_2, B_1, C)
        annotated_data.append(data_point)
    except:
        print(f"Error for question: {question_id} \n")
        errors.append(question_id)
    counter += 1
    print(f"Processed {counter} questions")
    print(f"\n{chat_A.budget()}")

print(f"Errors occured at {errors}") 
with open('NAME.json', 'w') as file_out: # INSERT HERE
    json.dump(annotated_data, file_out)

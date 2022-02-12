import numpy as np
import sys
from sklearn.metrics import accuracy_score as acsr

def get_answer(sent, sep_token="<sep>", end_token="</s>", classi_class=True):
    if classi_class:
        sent_ = sent.split(sep_token)[-1]
        class_ = sent.split(sep_token)[0]
        sent = sent_.split(end_token)[0]
        sent = sent.strip()
        return sent, class_
    sent = sent.split(end_token)[0]
    sent = sent.strip()
    return sent


def solve_problem2json(problem, i, model=None, tokenizer=None, classi_class=True):
    input_ids = tokenizer(problem,return_tensors='pt')['input_ids']
    output = model.generate(input_ids, max_length = 100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    sentence, class_ = get_answer(sentence) # 
    # print(problem.rstrip("<sys>"))
    # print('{')
    print(str(i+1) + ':')
    # print('=====')
    problem = problem.replace('"', "'")
    newsentence = sentence.replace('\n', '\n\n').replace('\n\n\n\n', '\n\n\n').replace('"', "'")
    print(f'  class: {class_}')
    print(f'  problem: "{problem}"')
    print(f'  code: "{newsentence}"')
    try:
        print("  answer:",end=' ')
        exec(sentence)
    except:
        print('error')
    print("")



def compute_metrics(eval_pred, tokenizer=None, homedir='~', classi_class=False):
    # print(help(eval_pred))
    logits, labels = eval_pred
    # print(logits[1])
    # print(logits, len(logits))
    # print(attn, len(attn))
    predictions = np.argmax(logits[0], axis=-1)
    print(get_answer(tokenizer.decode(predictions[0]))[0].replace('enter','\n'), classi_class)
    pred = []
    label = []
    A = sys.stdout
    B = sys.stdin
    sys.stdout = open(f"{homedir}/stdout.txt","w")
    sys.stdin = open(f"{homedir}/stdout.txt","r")
    count = 0 
    for i, j in zip(predictions, labels):
        count += 1
        i = tokenizer.decode(i).replace('enter', '\n')
        j = tokenizer.decode(j).replace('enter', '\n')
    
        try: exec(get_answer(i, classi_class=classi_class))
        except: print("error")
        try: pred.append(input())
        except: pred.append("bb")
        try: exec(get_answer(j, classi_class=classi_class))
        except: print("Error")
        try: label.append(input())
        except: label.append("ab")
    sys.stdout = A
    sys.stdin = B
    # print(pred, label)
    return {'accuracy':acsr(pred, label)}
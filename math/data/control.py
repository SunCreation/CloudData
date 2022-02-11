import pandas as pd
import re
import sys

A = sys.stdout
sys.stdout = open("double_slash.txt", 'w')
data = pd.read_csv("train.csv")
problem_num = []
code_num = []
count = 0 

for i, j in enumerate(zip(data["problem"], data["code"])):
    # nl2 = re.findall(r"[0-9]+",j[0])
    nl1 = re.findall("//",j[1])
    nl1.extend(re.findall(r"\sint",j[1]))
    nl1.extend(re.findall("round",j[1]))
    if nl1:
        count += 1
        print(i, j[0], j[1], sep='\n')
        print("")
sys.stdout = A
print(count)


import sys
import pandas as pd
# from datasets import load_dataset
# from transformers import AutoTokenizer
import time
data = pd.read_csv("~/Working/Python3/Deep/lms/CloudData/math/data/train.csv")
# A = sys.stdout
# sys.stdout = open("answer.txt", "w")
sys.stdin = open("answer.txt", "r")
count_ = 0
# a = "a=1\nb=1\nprint(a+b)"
# exec(a)
for i, j in enumerate(data["answer"]):
    a = input()
    if a==j:
        print(i, a, j)


# # print(a)
# for i, Q in zip(data["code"],data["answer"]):
#     하아 = i
#     # print(a)
#     # sys.stdout = A
#     exec(i)


    
#     count_ += 1
    
#     ans = input()
#     if ans!=Q: 
#         sys.stdout = A
#         time.sleep(1)
#         print(count_)
#         print(하아,A,ans)
#         sys.stdout = open("answer.txt", "a")
#         time.sleep(1)

# sys.stdout = A

# # print(a)
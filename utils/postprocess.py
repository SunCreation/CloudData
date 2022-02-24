import json
import pandas as pd
import re

def postprocess(ans):
    if re.search('\.',str(ans)):
        if int(str(ans).split(".")[1]) == 0:
            return str(int(float(ans)))
        else: return '%.2f' %float(ans)
    else: return str(ans)


filename = input('Enter only filename: ')
with open(f'{filename}.json', "r") as f:
    data = json.load(f)

# print(data["1"])

data = pd.DataFrame(data).transpose()
data['answer'] = data['answer'].apply(postprocess)
data = data.transpose()
data = data.to_dict()

data = json.dumps(data, indent=4)
with open(f"{filename}_postprocess.json", "w") as f:
    f.write(data)

print(data)
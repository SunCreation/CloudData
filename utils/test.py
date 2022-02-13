# import sys
# sys.stdout = open("yam.yaml", 'w')

# for i in range(100):
#     print(f'{i}: ')
#     print('  class^2: ',end='')
#     print(i**2)
#     print('  class^3: ',end='')
#     print(i**3)
import yaml
import pandas as pd

def check_yaml(filedir):
    cleanfile = []
    with open(filedir, "r") as f:
        for i in f.readlines():
            if i.startswith('error') or i.startswith('Error') or i.strip() == '': continue
            cleanfile.append(i)
    
    with open(filedir, "w") as f:
        for i in cleanfile:
            f.write(i)

check_yaml('yam.yaml')

with open("yam.yaml", 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
data = pd.DataFrame(data).transpose()
print(list(data['class^2']))
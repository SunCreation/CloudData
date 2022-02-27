import yaml
import pandas as pd

with open('inputdata.yaml', "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
# data = pd.DataFrame(data)


print(data)

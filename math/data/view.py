from koeda import EDA
import pandas as pd
data = pd.read_csv("train.csv")
eda = EDA(
    morpheme_analyzer="Okt", alpha_sr=0.0, alpha_ri=0.2, alpha_rs=0.0, prob_rd=0.0
)
for i in data["problem"][:5]:
    print(i)
    print('=====================================')
    print(eda(i))
    print("\n\n")
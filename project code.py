import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

#Import data
data = pd.read_csv (r'C:\Desktop\country_model_datasetV2(1).csv')
df = pd.DataFrame(data, columns = ['Model', "MBC", 'USD Profit']) 

##Notes 
#USD model as uniform distribution
#MBC taken as normal dist

# sort data frames by model type
df1 = df.groupby(['Model'])
mod_names = [name for name, Model in df1]
model_count = len(df1)
df_mean = pd.DataFrame(df1.mean()).reset_index()
df_std = pd.DataFrame(df1.std()).reset_index()

# montecarlo sim 
num_reps = 500
num_sims = 10000

final_stats = pd.DataFrame({"dummy": range(num_sims)})

##loop over models
for j in range(model_count):
    avg_profit = df_mean["USD Profit"][j]
    avg_MBC = df_mean["MBC"][j]
    std_profit = df_std["USD Profit"][j]
    std_MBC = df_std["MBC"][j]
    stats = []
    for i in range(num_sims):
        reps_net = np.random.uniform(avg_profit, std_profit, num_reps).round(2) 
        reps_MBC = np.random.normal(avg_MBC, std_MBC, num_reps).round(2)
        df_result = pd.DataFrame(index=range(num_reps),data = {"MBC": reps_MBC, "USD Profit": reps_net})
        df_result["Net Profit"]= df_result["USD Profit"]-df_result["MBC"] 
        stats.append(round(df_result["Net Profit"].mean(),2))
    stats1 = pd.DataFrame({mod_names[j]:stats})
    final_stats = final_stats.join(stats1)
final_stats =final_stats.drop(columns = ["dummy"])
print(final_stats)
print(final_stats.describe())

# will plot this for histogram (model)
for i in range(model_count):
    hist = final_stats[mod_names[i]].hist()
    plt.title(mod_names[i])
    plt.show()

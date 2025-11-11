from main import load_data, data_preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = data_preprocessing(load_data("heart-disease_data.csv"))
df = pd.DataFrame(data)
sns.set_theme(style="ticks")
sns.pairplot(df, hue="target")
plt.show()

import pandas as pd 
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/code/master/streamlit/part3/penguins_cleaned.csv")
df.to_csv("penguins_cleaned.csv")
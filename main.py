import pandas as pd

path = "data/student_health_data.csv"

df = pd.read_csv(path)
print(df.head(10))
print(df.info())

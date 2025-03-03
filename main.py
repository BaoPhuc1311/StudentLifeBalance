import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

path = "data/student_health_data.csv"

df = pd.read_csv(path)

print(df.info())
print(df.isnull().sum())

nominal_col = ["Gender"]
ordinal_col = ["Physical_Activity", "Sleep_Quality", "Mood", "Health_Risk_Level"]

nominal_encoder = OneHotEncoder(drop="first", sparse_output=False)
df[nominal_col] = nominal_encoder.fit_transform(df[nominal_col]).astype(int)

ordinal_encoder = LabelEncoder()
for col in ordinal_col:
    df[col] = ordinal_encoder.fit_transform(df[col])

print(df.head(10))

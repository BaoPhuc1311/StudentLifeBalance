import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

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

X = df.drop(columns=["Health_Risk_Level"])
y = df["Health_Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
report = classification_report(y_test, y_predict)

print(f"Accuracy Score: {accuracy}")
print(f"Classification Report:\n {report}")

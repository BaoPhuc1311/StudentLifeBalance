import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from imblearn.over_sampling import SMOTE

path = "./data/student_health_data.csv"
df = pd.read_csv(path)

print(df.info())

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

df["Health_Risk_Level"].value_counts().sort_index()

smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

y_resampled.value_counts().sort_index()

plt.figure(figsize=(7, 5))
sns.countplot(x=y_resampled, palette='coolwarm')
plt.title("Phân phối của Health Risk Level")
plt.xlabel("Health Risk Level")
plt.ylabel("Số lượng sinh viên")
plt.savefig("./images/histogram.png")  
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(class_weight="balanced", criterion="gini", max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_true = y_test
y_predict = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
report = classification_report(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"Accuracy Score: {accuracy}")
print(f"Classification Report:\n {report}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R² Score: {r2}")

cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("./images/confusion_matrix.png")  
plt.show()

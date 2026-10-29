
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error

# 1. Create sample elderly health dataset
df = pd.DataFrame({
    "Age": [60, 62, 65, 68, 70, 72, 75, 78, 80, 82],
    "HeartRate": [78, 76, 80, 75, 82, 79, 84, 80, 86, 83],
    "SystolicBP": [120, 125, 130, 128, 135, 132, 140, 138, 145, 142],
    "SpO2": [98, 97, 98, 96, 97, 96, 95, 96, 94, 95]
})

# 2. NumPy operations
ages = df["Age"].to_numpy()

print("Ages:", ages)
print("First 3 ages:", ages[:3])
print("Ages above 70:", ages[ages > 70])
print("Reshaped ages:\n", ages.reshape(2, 5))
print("Ages + 1:", ages + 1)

# 3. Calculate a simple illustrative health score
# These rules are for a programming demonstration,
# not a clinical health assessment.
score = np.full(len(df), 100.0)

score -= np.where(
    (df["HeartRate"] < 60) | (df["HeartRate"] > 100), 15, 0
)
score -= np.where(
    (df["SystolicBP"] < 90) | (df["SystolicBP"] > 140), 15, 0
)
score -= np.where(df["SpO2"] < 95, 20, 0)

df["HealthScore"] = score

# 4. Dataset inspection
print("\nDataset:\n", df)
print("\nDataset information:")
df.info()
print("\nDescriptive statistics:\n", df.describe())

# 5. Data preprocessing
print("\nMissing values:\n", df.isnull().sum())
df = df.drop_duplicates()
df["SpO2"] = df["SpO2"].fillna(df["SpO2"].mean())

# 6. Filtering and sorting
print("\nHealth scores below 100:")
print(df[df["HealthScore"] < 100])

print("\nSorted by health score:")
print(df.sort_values("HealthScore", ascending=False))

# 7. Grouping and aggregation
df["HealthCategory"] = np.where(
    df["HealthScore"] >= 80, "Higher Score", "Lower Score"
)

print("\nCategory summary:")
print(df.groupby("HealthCategory")["HealthScore"].agg(["count", "mean"]))

# 8. Statistical operations
print("\nMean health score:", np.mean(df["HealthScore"]))
print("Median health score:", np.median(df["HealthScore"]))
print("Variance:", np.var(df["HealthScore"]))
print("Standard deviation:", np.std(df["HealthScore"]))
print("\nCorrelation:\n", df[["Age", "HeartRate",
                              "SystolicBP", "SpO2",
                              "HealthScore"]].corr())

# 9. Line plot
plt.plot(df["Age"], df["HealthScore"], marker="o")
plt.xlabel("Age")
plt.ylabel("Illustrative Health Score")
plt.title("Age vs Health Score")
plt.show()

# 10. Bar chart
plt.bar(df["Age"], df["HealthScore"])
plt.xlabel("Age")
plt.ylabel("Illustrative Health Score")
plt.title("Health Score by Age")
plt.show()

# 11. Histogram
plt.hist(df["HealthScore"], bins=5, edgecolor="black")
plt.xlabel("Health Score")
plt.ylabel("Frequency")
plt.title("Health Score Distribution")
plt.show()

# 12. Scatter plot
plt.scatter(df["Age"], df["SystolicBP"])
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure")
plt.title("Age vs Systolic BP")
plt.show()

# 13. Correlation heatmap
sns.heatmap(
    df[["Age", "HeartRate", "SystolicBP", "SpO2",
        "HealthScore"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Health Data Correlation")
plt.show()

# 14. Predict age using health readings
X = df[["HeartRate", "SystolicBP", "SpO2", "HealthScore"]]
y = df["Age"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nActual ages:", list(y_test))
print("Predicted ages:", np.round(predictions, 1))
print("Mean Absolute Error:",
      round(mean_absolute_error(y_test, predictions), 2), "years")

# Enter readings to estimate age
print("\nEnter health readings to estimate age:")
heart = float(input("Heart rate (bpm): "))
bp = float(input("Systolic blood pressure (mmHg): "))
spo2 = float(input("SpO2 (%): "))

new_score = 100
if heart < 60 or heart > 100:
    new_score -= 15
if bp < 90 or bp > 140:
    new_score -= 15
if spo2 < 95:
    new_score -= 20

new_person = pd.DataFrame({
    "HeartRate": [heart],
    "SystolicBP": [bp],
    "SpO2": [spo2],
    "HealthScore": [new_score]
})

estimated_age = model.predict(new_person)
print("Estimated age:", round(estimated_age[0], 1), "years")

# 15. Basic health score classification
df["ScoreClass"] = np.where(df["HealthScore"] >= 80, 1, 0)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X, df["ScoreClass"])

predicted_class = classifier.predict(new_person)
print("Score category:",
      "Higher Score" if predicted_class[0] == 1 else "Lower Score")

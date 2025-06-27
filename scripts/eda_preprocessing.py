import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../data/transactions.csv')

# print(df.head()) # prints first five rows of the dataframe

# print(df.info()) # prints dataset info

# print("Missing transactions:", df.isnull().sum()) # prints total number of missing values

# print("Class Distribution:")
# print(df['Class'].value_counts())

# sns.countplot(x='Class', data=df)
# plt.title("Fraud(1) & Non-Fraud(0)")
# plt.show()

# print("Amount Statistics:")
# print(df['Amount'].describe())

corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', linewidth=0.5)
plt.title("Correlation Heatmap")
plt.show()

df.to_csv('../data/processed_transactions.csv', index=False)
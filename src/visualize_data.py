print("Name: Arpita")
print("URN: 2581127")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/train.csv")

# Count genuine and fraudulent transactions
fraud_count = data["is_fraud"].value_counts()

# Create bar graph
plt.figure(figsize=(8, 5))

sns.barplot(
    x=fraud_count.index.astype(str),
    y=fraud_count.values
)

plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Transaction Type (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.savefig("results/transaction_distribution.png")
plt.show()

print("Graph saved successfully!")
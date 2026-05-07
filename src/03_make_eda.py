import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("reports").mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv("data/processed/train_processed.csv")
valid_df = pd.read_csv("data/processed/valid_processed.csv")

# 길이 컬럼 생성
train_df["length"] = train_df["comments"].astype(str).str.len()
valid_df["length"] = valid_df["comments"].astype(str).str.len()

# 1. 라벨 분포
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=train_df)
plt.title("Train Label Distribution")
plt.xlabel("label (0=normal, 1=malicious)")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("reports/train_label_distribution.png")
plt.close()

# 2. 댓글 길이 분포
plt.figure(figsize=(8, 4))
sns.histplot(train_df["length"], bins=50, kde=True)
plt.title("Train Comment Length Distribution")
plt.xlabel("comment length")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("reports/train_length_distribution.png")
plt.close()

print("EDA 그래프 저장 완료")
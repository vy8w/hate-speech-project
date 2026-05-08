import pandas as pd
from pathlib import Path

# 저장 폴더 생성
Path("data/raw").mkdir(parents=True, exist_ok=True)

# GitHub raw URL에서 직접 읽기
train_url = "https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/train.tsv"
dev_url = "https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/dev.tsv"

train_df = pd.read_csv(train_url, sep="\t")
dev_df = pd.read_csv(dev_url, sep="\t")

# 원본 저장
train_df.to_csv("data/raw/train_raw.csv", index=False)
dev_df.to_csv("data/raw/dev_raw.csv", index=False)

print("train shape:", train_df.shape)
print("dev shape:", dev_df.shape)
print("columns:", train_df.columns.tolist())
print(train_df.head())
from Korpora import Korpora
import pandas as pd
from pathlib import Path

# 저장 폴더 생성
Path("data/raw").mkdir(parents=True, exist_ok=True)

# 데이터셋 로드
corpus = Korpora.load("korean_hate_speech")

train_df = pd.DataFrame(corpus.train)
dev_df = pd.DataFrame(corpus.dev)

# 원본 저장
train_df.to_csv("data/raw/train_raw.csv", index=False)
dev_df.to_csv("data/raw/dev_raw.csv", index=False)

print("train shape:", train_df.shape)
print("dev shape:", dev_df.shape)
print("columns:", train_df.columns.tolist())
print(train_df.head())
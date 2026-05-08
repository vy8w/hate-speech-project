import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(parents=True, exist_ok=True)

# 원본 데이터 불러오기
train_df = pd.read_csv("data/raw/train_raw.csv")
dev_df = pd.read_csv("data/raw/dev_raw.csv")

# 컬럼 이름 확인 후 text -> comments로 통일
if "text" in train_df.columns:
    train_df = train_df.rename(columns={"text": "comments"})
if "text" in dev_df.columns:
    dev_df = dev_df.rename(columns={"text": "comments"})

# 라벨 매핑 함수
def map_label(x):
    if x in ["hate", "offensive"]:
        return 1
    return 0

train_df["label"] = train_df["hate"].apply(map_label)
dev_df["label"] = dev_df["hate"].apply(map_label)

# 필요한 컬럼만 선택
use_cols = ["comments", "hate", "label"]
train_df = train_df[use_cols].copy()
dev_df = dev_df[use_cols].copy()

# 전처리 함수
def clean_df(df):
    df = df.copy()

    # 결측 제거
    df = df.dropna(subset=["comments"])

    # 문자열화 + 공백 제거
    df["comments"] = df["comments"].astype(str).str.strip()

    # 빈 문자열 제거
    df = df[df["comments"] != ""]

    # 너무 짧은 댓글 제거 (길이 3 미만 제거)
    df = df[df["comments"].str.len() >= 3]

    # 중복 제거
    df = df.drop_duplicates(subset=["comments"])

    return df

train_df = clean_df(train_df)
dev_df = clean_df(dev_df)

# train + dev 합치기
all_df = pd.concat([train_df, dev_df], ignore_index=True)

# 최종 train / valid 분할
train_processed, valid_processed = train_test_split(
    all_df,
    test_size=0.2,
    stratify=all_df["label"],
    random_state=42
)

# 저장
train_processed.to_csv("data/processed/train_processed.csv", index=False)
valid_processed.to_csv("data/processed/valid_processed.csv", index=False)

# 간단한 통계 저장
summary = {
    "raw_train_rows": [len(pd.read_csv("data/raw/train_raw.csv"))],
    "raw_dev_rows": [len(pd.read_csv("data/raw/dev_raw.csv"))],
    "processed_train_rows": [len(train_processed)],
    "processed_valid_rows": [len(valid_processed)],
    "train_label_0": [(train_processed["label"] == 0).sum()],
    "train_label_1": [(train_processed["label"] == 1).sum()],
    "valid_label_0": [(valid_processed["label"] == 0).sum()],
    "valid_label_1": [(valid_processed["label"] == 1).sum()],
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv("reports/preprocessing_summary.csv", index=False)

print("전처리 완료")
print("train_processed shape:", train_processed.shape)
print("valid_processed shape:", valid_processed.shape)
print(train_processed["label"].value_counts())
print(valid_processed["label"].value_counts())
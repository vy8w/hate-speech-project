import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

Path("reports").mkdir(parents=True, exist_ok=True)

# 데이터 불러오기
train_df = pd.read_csv("data/processed/train_processed.csv")
valid_df = pd.read_csv("data/processed/valid_processed.csv")

X_train = train_df["comments"]
y_train = train_df["label"]
X_valid = valid_df["comments"]
y_valid = valid_df["label"]

# TF-IDF 벡터화
tfidf = TfidfVectorizer(max_features=10000)
X_train_vec = tfidf.fit_transform(X_train)
X_valid_vec = tfidf.transform(X_valid)

# 모델 학습 및 평가 함수
def evaluate_model(model, name):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_valid_vec)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {accuracy_score(y_valid, y_pred):.4f}")
    print(f"Precision: {precision_score(y_valid, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_valid, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_valid, y_pred):.4f}")
    print(classification_report(y_valid, y_pred, target_names=["정상", "악성"]))

    # Confusion Matrix 저장
    cm = confusion_matrix(y_valid, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["정상", "악성"],
                yticklabels=["정상", "악성"])
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("실제")
    plt.xlabel("예측")
    plt.tight_layout()
    plt.savefig(f"reports/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()
    print(f"Confusion Matrix 저장 완료: reports/{name}_confusion_matrix.png")

    # 예측 결과 CSV 저장 (오류 분석용)
    result_df = valid_df.copy()
    result_df["predicted"] = y_pred
    result_df["correct"] = (result_df["predicted"] == result_df["label"])
    result_df.to_csv(f"reports/{name.replace(' ', '_')}_predictions.csv", index=False)
    print(f"예측 결과 저장 완료: reports/{name.replace(' ', '_')}_predictions.csv")

# 로지스틱 회귀
evaluate_model(LogisticRegression(max_iter=1000), "Logistic Regression")

# 선형 SVM
evaluate_model(LinearSVC(max_iter=1000), "Linear SVM")
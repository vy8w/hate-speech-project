from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_NAME = "klue/bert-base"
MAX_LENGTH = 128
OUTPUT_DIR = Path("model/klue_bert_lora_finetuned")
REPORT_DIR = Path("reports")


# Trainer가 사용할 수 있도록 댓글과 라벨을 PyTorch Dataset 형태로 변환한다.
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# 검증 데이터 기준으로 성능 비교에 필요한 지표를 계산한다.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


# LoRA 모델의 예측 결과를 confusion matrix 이미지로 저장한다.
def save_confusion_matrix(y_true, y_pred, output_path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["정상", "악성"],
        yticklabels=["정상", "악성"],
    )
    plt.title("KLUE BERT LoRA Confusion Matrix")
    plt.xlabel("예측")
    plt.ylabel("실제")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    set_seed(42)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 전처리 완료 데이터를 그대로 사용한다.
    train_df = pd.read_csv("data/processed/train_processed.csv")
    valid_df = pd.read_csv("data/processed/valid_processed.csv")

    train_texts = train_df["comments"].astype(str).tolist()
    valid_texts = valid_df["comments"].astype(str).tolist()
    train_labels = train_df["label"].astype(int).tolist()
    valid_labels = valid_df["label"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # KLUE-BERT의 attention query/value 모듈에 LoRA adapter를 추가한다.
    # 전체 파라미터를 모두 학습하지 않고, 작은 adapter 파라미터만 학습하는 방식이다.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer)
    valid_dataset = HateSpeechDataset(valid_texts, valid_labels, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="reports/logs_lora",
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # LoRA adapter만 학습해 악성 댓글 이진 분류 태스크에 맞춘다.
    trainer.train()

    predictions = trainer.predict(valid_dataset)
    y_pred = predictions.predictions.argmax(axis=-1)
    y_true = predictions.label_ids

    metrics = {
        "model": MODEL_NAME,
        "method": "LoRA",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    print("\n===== KLUE BERT LoRA Fine-tuning Result =====")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(classification_report(y_true, y_pred, target_names=["정상", "악성"]))

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "KLUE_BERT_LoRA_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result_df = valid_df.copy()
    result_df["predicted"] = y_pred
    result_df["correct"] = result_df["predicted"] == result_df["label"]
    result_df.to_csv(
        REPORT_DIR / "KLUE_BERT_LoRA_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        REPORT_DIR / "KLUE_BERT_LoRA_confusion_matrix.png",
    )

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n저장 완료")
    print(f"- {REPORT_DIR / 'KLUE_BERT_LoRA_metrics.csv'}")
    print(f"- {REPORT_DIR / 'KLUE_BERT_LoRA_predictions.csv'}")
    print(f"- {REPORT_DIR / 'KLUE_BERT_LoRA_confusion_matrix.png'}")
    print(f"- {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

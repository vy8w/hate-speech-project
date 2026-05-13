from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPORT_DIR = Path("reports")
PRED_PATH = REPORT_DIR / "KLUE_BERT_LoRA_predictions.csv"


# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────────
def set_korean_font():
    font_keywords = ["nanum", "gothic", "cjk", "hangul", "malgun"]
    candidates = [
        f for f in fm.findSystemFonts()
        if any(k in f.lower() for k in font_keywords)
    ]
    if candidates:
        fm.fontManager.addfont(candidates[0])
        plt.rcParams["font.family"] = fm.FontProperties(fname=candidates[0]).get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


# ── 오류 유형 분류 ───────────────────────────────────────────────────────────────
def classify_error(text: str) -> str:
    """
    댓글 텍스트를 받아 오류 유형 문자열을 반환한다.

    분류 기준:
    - 짧은 댓글      : 길이 15자 이하 → 정보 부족으로 인한 오분류
    - 속어/신조어 혼동 : 긍정 은어(쩌네, 지렸다 등)를 악성으로 오인
    - 완곡한 비판 표현 : 공손한 어조의 비판을 정상으로 오인
    - 외모 관련 표현  : 외모 언급 자체를 악성으로 오인
    - 문맥 의존적 표현 : 앞뒤 맥락 없이는 판단이 어려운 경우
    """
    text = str(text)

    if len(text) <= 15:
        return "짧은 댓글"

    slang_positive = ["쩌네", "지렸다", "미쳤다", "개꿀", "개잼", "ㅋㅋ", "짱", "존맛"]
    if any(w in text for w in slang_positive):
        return "속어/신조어 혼동"

    polite_critic = ["죄송", "아닌것같", "아닌 것 같", "좀", "하차", "그만", "자중"]
    if any(w in text for w in polite_critic):
        return "완곡한 비판 표현"

    body_words = ["얼굴", "몸", "살", "성형", "가슴", "허벅지", "키"]
    if any(w in text for w in body_words):
        return "외모 관련 표현"

    return "문맥 의존적 표현"


# ── 오류 분류 및 통계 출력 ────────────────────────────────────────────────────────
def analyze_errors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = df[df["correct"] == False].copy()

    fp = errors[(errors["label"] == 0) & (errors["predicted"] == 1)].copy()
    fn = errors[(errors["label"] == 1) & (errors["predicted"] == 0)].copy()

    fp["error_type"] = fp["comments"].apply(classify_error)
    fn["error_type"] = fn["comments"].apply(classify_error)

    total = len(df)
    print("=" * 55)
    print("              오류 분석 결과")
    print("=" * 55)
    print(f"  전체 샘플 수     : {total:,}건")
    print(f"  정답 수          : {(df['correct'] == True).sum():,}건")
    print(f"  오류 수          : {len(errors):,}건  ({len(errors)/total*100:.1f}%)")
    print(f"  FP (정상→악성)   : {len(fp):,}건")
    print(f"  FN (악성→정상)   : {len(fn):,}건")
    print("-" * 55)

    print("\n[FP 오류 유형 분포 — 정상인데 악성으로 오분류]")
    print(fp["error_type"].value_counts().to_string())

    print("\n[FN 오류 유형 분포 — 악성인데 정상으로 오분류]")
    print(fn["error_type"].value_counts().to_string())

    print("\n[FP 대표 사례 10건]")
    for _, row in fp.head(10).iterrows():
        print(f"  [{row['hate']}] {row['comments'][:60]}")

    print("\n[FN 대표 사례 10건]")
    for _, row in fn.head(10).iterrows():
        print(f"  [{row['hate']}] {row['comments'][:60]}")

    return fp, fn


# ── 시각화 1: FP / FN 오류 유형 막대 그래프 ─────────────────────────────────────
def plot_error_bar(fp: pd.DataFrame, fn: pd.DataFrame, save_path: Path) -> None:
    fp_counts = fp["error_type"].value_counts()
    fn_counts = fn["error_type"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(fp_counts.index, fp_counts.values, color="#E07B7B", edgecolor="white")
    axes[0].set_title("FP 오류 유형\n(정상 → 악성 오분류)", fontsize=13)
    axes[0].set_ylabel("건수")
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(fp_counts.values):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontsize=10)

    axes[1].bar(fn_counts.index, fn_counts.values, color="#7BA8E0", edgecolor="white")
    axes[1].set_title("FN 오류 유형\n(악성 → 정상 오분류)", fontsize=13)
    axes[1].set_ylabel("건수")
    axes[1].tick_params(axis="x", rotation=25)
    for i, v in enumerate(fn_counts.values):
        axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=10)

    plt.suptitle("KLUE-BERT LoRA 오류 유형 분포", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"막대 그래프 저장 완료: {save_path}")


# ── 시각화 2: 전체 오류 유형 파이차트 ────────────────────────────────────────────
def plot_error_pie(fp: pd.DataFrame, fn: pd.DataFrame, save_path: Path) -> None:
    all_errors = pd.concat([fp, fn])
    counts = all_errors["error_type"].value_counts()

    colors = ["#5B7FD4", "#E07B7B", "#7BC99E", "#F5C06A", "#B57BE0"]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors[:len(counts)],
        startangle=140,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.set_title("오류 유형 전체 분포 (FP + FN)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"파이차트 저장 완료: {save_path}")


# ── 시각화 3: 오류 유형별 댓글 길이 분포 박스플롯 ────────────────────────────────
def plot_length_by_error_type(fp: pd.DataFrame, fn: pd.DataFrame, save_path: Path) -> None:
    all_errors = pd.concat([fp, fn]).copy()
    all_errors["length"] = all_errors["comments"].str.len()

    order = all_errors.groupby("error_type")["length"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=all_errors,
        x="error_type",
        y="length",
        order=order,
        palette="pastel",
        ax=ax,
    )
    ax.set_title("오류 유형별 댓글 길이 분포", fontsize=13, fontweight="bold")
    ax.set_xlabel("오류 유형")
    ax.set_ylabel("댓글 길이 (글자 수)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"박스플롯 저장 완료: {save_path}")


# ── CSV 저장 ─────────────────────────────────────────────────────────────────────
def save_error_csv(fp: pd.DataFrame, fn: pd.DataFrame, save_dir: Path) -> None:
    fp.to_csv(save_dir / "error_fp_cases.csv", index=False, encoding="utf-8-sig")
    fn.to_csv(save_dir / "error_fn_cases.csv", index=False, encoding="utf-8-sig")
    print(f"FP 오류 CSV 저장 완료: {save_dir / 'error_fp_cases.csv'}")
    print(f"FN 오류 CSV 저장 완료: {save_dir / 'error_fn_cases.csv'}")


# ── 메인 ─────────────────────────────────────────────────────────────────────────
def main():
    set_korean_font()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PRED_PATH)

    fp, fn = analyze_errors(df)

    plot_error_bar(fp, fn,  REPORT_DIR / "error_type_bar.png")
    plot_error_pie(fp, fn,  REPORT_DIR / "error_type_pie.png")
    plot_length_by_error_type(fp, fn, REPORT_DIR / "error_length_boxplot.png")

    save_error_csv(fp, fn, REPORT_DIR)

    print("\n오류 분석 완료")


if __name__ == "__main__":
    main()

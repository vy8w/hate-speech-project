# Korean Hate Speech Project - Data Preparation

## 데이터셋
- Korean Hate Speech Dataset
- GitHub: https://github.com/kocohub/korean-hate-speech
- Korpora: https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/korean_hate_speech.html

## 역할
- 데이터셋 준비 및 전처리: 표정인

## 전처리 기준
- hate/offensive -> 1
- none -> 0
- 결측 제거
- 공백 제거
- 길이 3 미만 댓글 제거
- 중복 댓글 제거

## 실행 순서
1. python src/01_load_data.py
2. python src/02_preprocess_data.py
3. python src/03_make_eda.py

## 결과 파일
- data/processed/train_processed.csv
- data/processed/valid_processed.csv
- reports/preprocessing_summary.csv
- reports/train_label_distribution.png
- reports/train_length_distribution.png
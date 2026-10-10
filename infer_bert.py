"""
infer_bert.py

載入 train_bert.py 儲存的最佳 BERT 8 類情緒分類模型，
對新的中文句子進行情緒分類。

label 名稱已存在模型的 config 裡（id2label），
所以預測結果直接是中文，例如「開心語調」，不用另外讀 labels.json。
"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# =============================================================================
# 1. 推論設定
# =============================================================================

MODEL_DIR = Path("output/bert_emotion_clf")

# 可以直接修改這個 list 測試不同句子。
TEST_TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。",
]


# =============================================================================
# 2. 主流程
# =============================================================================


def main():

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到 fine-tuned model：{MODEL_DIR.resolve()}\n"
            "請先執行 train_bert.py。"
        )

    # -------------------------------------------------------------------------
    # Step 1：載入 tokenizer
    # -------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # -------------------------------------------------------------------------
    # Step 2：載入 sequence classifier
    # -------------------------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # -------------------------------------------------------------------------
    # Step 3：決定 inference device
    # -------------------------------------------------------------------------

    # pipeline：
    #
    #     device=0
    #
    # 代表使用第 0 張 CUDA GPU。
    #
    #     device=-1
    #
    # 代表使用 CPU。
    device = 0 if torch.cuda.is_available() else -1

    print(
        "Inference device：" + (torch.cuda.get_device_name(0) if device == 0 else "CPU")
    )

    # -------------------------------------------------------------------------
    # Step 4：建立 Hugging Face pipeline
    # -------------------------------------------------------------------------

    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Step 5：執行分類
    # -------------------------------------------------------------------------

    results = classifier(
        TEST_TEXTS,
        truncation=True,
        max_length=512,
    )

    # -------------------------------------------------------------------------
    # Step 6：顯示結果
    # -------------------------------------------------------------------------

    for text, result in zip(
        TEST_TEXTS,
        results,
    ):
        print(f"{text} => {result['label']} ({result['score']:.4f})")


if __name__ == "__main__":
    main()

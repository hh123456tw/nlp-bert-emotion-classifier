"""
prepare_data.py

把原始 data.csv 轉成 BERT 8 類情緒分類訓練所需的 train / validation JSONL。

原始資料是 CSV，第一列是標題，之後一行一筆：

    text,emotion
    你要不要去吃午餐？,平淡語氣
    誒誒誒！我甄選上了！,開心語調

emotion 是中文標籤，共 8 種，微調時必須轉成整數 0–7。

這支程式做的事情：

    data/data.csv
        ↓
    讀取文字與 emotion
        ↓
    標籤排序後編號 0–7（label2id）
        ↓
    以 label 做 stratified train / validation split
        ↓
    data/bert/train.jsonl
    data/bert/valid.jsonl
    data/bert/labels.json（label2id 對照表）

為什麼要先獨立做資料切分？

因為之後 train_bert.py 每次執行時，都直接讀取固定的 train / validation 檔案，
比較不同模型或超參數時，不會因為每次重新亂數切分而改變資料集。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

# csv：
# 原始 data.csv 以逗號分隔，第一列是標題（text,emotion）。
# 使用 csv.DictReader 可以直接用欄位名稱取值，並自動略過標題列。
import csv

# json：
# 用來把 Python dict 寫成 JSONL。
import json

# Counter：
# 用來統計每個 label 各有多少筆。
from collections import Counter

# Path：
# 用來處理輸入與輸出路徑。
from pathlib import Path

# train_test_split：
# 用來切 training / validation。
#
# 這裡使用 stratify=labels，
# 讓 train 與 validation 裡 8 個類別的比例盡量接近原始資料。
from sklearn.model_selection import train_test_split

# =============================================================================
# 2. 資料設定
# =============================================================================

# 原始資料檔。
INPUT_FILE = Path("data/data.csv")

# 輸出資料夾。
OUTPUT_DIR = Path("data/bert")

# Validation 比例。
#
# 0.2 = 20%
#
# 所以大約：
#     80% → training
#     20% → validation
VALID_RATIO = 0.2

# 固定 random seed。
#
# 讓每次執行都得到相同的 train / validation split，
# 方便重現實驗。
SEED = 42


# =============================================================================
# 3. 輔助函式
# =============================================================================


def load_reviews(path: Path):
    """
    讀取 data.csv（逗號分隔，第一列是標題 text,emotion）。

    回傳：
        texts  : list[str]   每一筆的文字
        labels : list[str]   每一筆的情緒標籤（中文，例如 "開心語調"）

    注意：這裡的 labels 還是中文，沒有轉成整數。
    因為要讀完全部資料，才知道一共有哪些標籤，
    所以「文字 → 整數」的轉換放在 main() 裡做。
    """

    texts = []
    labels = []

    # 記錄被略過的空句子行號。
    #
    # 這些 row 雖然可能帶有 label，
    # 但沒有任何文字內容，
    # 對 text classification 沒有可學習的輸入資訊。
    skipped_empty_rows = []

    # newline="" 是 csv 模組官方建議的開檔方式，
    # 可以讓 csv 模組正確處理換行。
    with path.open("r", encoding="utf-8", newline="") as file:

        # delimiter=","：
        #     使用逗號分隔欄位。
        #
        # quotechar='"'：
        #     文字欄位外層的雙引號由 csv 模組處理，
        #     不會把雙引號本身留下來當成句子內容。
        reader = csv.DictReader(
            file,
            delimiter=",",
            quotechar='"',
        )

        for line_number, row in enumerate(reader, start=1):

            # 空白行直接略過。
            if not row:
                continue

            text = row["text"].strip()

            # -------------------------------------------------------------
            # 空句子直接略過
            # -------------------------------------------------------------
            #
            # 原始資料中如果存在：
            #
            #     ,開心語調
            #
            # 也就是：
            #
            #     text    = 空字串
            #     emotion = 開心語調
            #
            # 這種資料沒有任何文字資訊可以讓 BERT 學習，
            # 因此不應該拿去做訓練。
            #
            # 與其讓整個資料前處理流程因一筆空資料中止，
            # 這裡選擇：
            #
            #     skip
            #
            # 並記錄被略過的行號，最後統一顯示統計資訊。
            if not text:
                skipped_empty_rows.append(line_number)
                continue

            label = row["emotion"].strip()

            texts.append(text)
            labels.append(label)

    if not texts:
        raise RuntimeError(f"{path} 沒有讀到任何有效資料。")

    # 顯示空句子清理結果。
    #
    # 這不是 training error，
    # 而是資料清理的一部分。
    if skipped_empty_rows:
        print(
            "已略過空句子："
            f"{len(skipped_empty_rows)} 筆；"
            f"原始行號={skipped_empty_rows}"
        )

    return texts, labels


def save_jsonl(texts, labels, path: Path):
    """
    將文字與標籤寫成 JSONL。

    每一行格式：

        {"sentences": "這間飯店很好", "labels": 1}

    為了和原 notebook 的欄位名稱保持一致，
    這裡繼續使用：
        sentences
        labels
    """

    with path.open("w", encoding="utf-8") as file:
        for text, label in zip(texts, labels):

            row = {
                "sentences": text,
                "labels": label,
            }

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_label_distribution(name, labels):
    """
    顯示每個 label 的資料量與比例。
    """

    counts = Counter(labels)
    total = len(labels)

    print(f"{name}：{total} 筆")

    for label in sorted(counts):
        count = counts[label]
        ratio = count / total

        print(f"  label={label}: " f"{count} 筆 " f"({ratio:.2%})")


# =============================================================================
# 4. 主流程
# =============================================================================


def main():

    # -------------------------------------------------------------------------
    # Step 1：確認原始資料存在
    # -------------------------------------------------------------------------

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"找不到原始資料：{INPUT_FILE.resolve()}")

    # -------------------------------------------------------------------------
    # Step 2：讀取 data.csv，並把中文標籤轉成整數
    # -------------------------------------------------------------------------

    texts, labels = load_reviews(INPUT_FILE)
    labels_name = sorted(set(labels))
    label2id = {}

    for i, name in enumerate(labels_name):
        label2id[name] = i

    labels = [label2id[x] for x in labels]

    print("=" * 72)
    print_label_distribution("全部資料", labels)

    # -------------------------------------------------------------------------
    # Step 3：切 train / validation
    # -------------------------------------------------------------------------

    # stratify=labels 很重要。
    #
    # 你的資料不是完全 50 / 50，
    # 所以如果只做普通亂數切分，
    # train 與 validation 的 label 比例可能出現額外偏差。
    #
    # stratify 會讓兩邊盡量維持相似的 class distribution。
    (
        train_texts,
        valid_texts,
        train_labels,
        valid_labels,
    ) = train_test_split(
        texts,
        labels,
        test_size=VALID_RATIO,
        random_state=SEED,
        stratify=labels,
    )

    # -------------------------------------------------------------------------
    # Step 4：建立輸出資料夾
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    # -------------------------------------------------------------------------
    # Step 5：儲存 JSONL
    # -------------------------------------------------------------------------

    save_jsonl(
        train_texts,
        train_labels,
        train_path,
    )

    save_jsonl(
        valid_texts,
        valid_labels,
        valid_path,
    )
    labels_path = OUTPUT_DIR / "labels.json"
    with labels_path.open("w", encoding="utf-8") as file:
        json.dump(label2id, file, ensure_ascii=False, indent=2)
    # -------------------------------------------------------------------------
    # Step 6：顯示結果
    # -------------------------------------------------------------------------

    print_label_distribution(
        "Training",
        train_labels,
    )

    print_label_distribution(
        "Validation",
        valid_labels,
    )

    print(f"Training JSONL：{train_path}")
    print(f"Validation JSONL：{valid_path}")
    print(f"Labels JSON：{labels_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

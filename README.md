# 中文句子情緒分類

## 訓練資料來源
- [Datasets:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)

## 基礎模型
- [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

## 安裝套件
<!-- TODO: 用 pip list 填入實際版本號 -->
- torch (TODO)
- transformers (TODO)
- datasets (TODO)
- evaluate (TODO)
- accelerate (TODO)
- scikit-learn (TODO)

## 說明
<!-- TODO: 模型用途、分成幾類、label 對應的整數 -->

### 預測結果
<!-- TODO: 貼上實際執行結果,score 取小數點後第 4 位 -->
| 文字 | 預測 | score |
|---|---|---|
| 我每天都能跟她一起上學,我好開心! | TODO | TODO |
| 最好的朋友要離開臺灣了,以後可能不容易再見面... | TODO | TODO |
| 我覺得我快不行了... | TODO | TODO |
| 剛剛收到研究所錄取的通知書! | TODO | TODO |
| 今年的冬天好像比較晚來。 | TODO | TODO |

## 執行方法
```
python XXX.py
```

## 成果
<!-- TODO: 執行截圖 -->

## 專案結構
```
output/   微調後的 best model(不含 checkpoint-*)
*.py      訓練與預測程式
README.md
```

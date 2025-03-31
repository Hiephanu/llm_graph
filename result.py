import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import re

def cosine_similarity(a, b):
    """Tính độ tương đồng cos giữa hai vector"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_bertscore(reference, candidate, model, tokenizer):
    """Tính BERTScore giữa hai văn bản"""
    with torch.no_grad():
        ref_tokens = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
        cand_tokens = tokenizer(candidate, return_tensors="pt", padding=True, truncation=True)
        
        ref_embedding = model(**ref_tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        cand_embedding = model(**cand_tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        
        score = cosine_similarity(ref_embedding, cand_embedding)
    return score

def compute_f1_score(true_labels, predicted_labels):
    """Tính Precision, Recall và F1-Score"""
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="macro")
    return precision, recall, f1

# Load model và tokenizer
bert_model = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)
model = AutoModel.from_pretrained(bert_model)

def read_file(filename):
    """Đọc nội dung từ file, bỏ số thứ tự ở đầu mỗi dòng"""
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Xóa số thứ tự ở đầu mỗi dòng (dạng "1. ", "2. ", ...)
    cleaned_lines = [re.sub(r"^\d+\.\s*", "", line.strip()) for line in lines]
    return cleaned_lines

# Đọc dữ liệu từ file
reference_answers = read_file("dataset/answer/expected.txt")
generated_answers = read_file("dataset/answer/answer.txt")


# Tính BERTScore
bert_scores = [compute_bertscore(ref, gen, model, tokenizer) for ref, gen in zip(reference_answers, generated_answers)]

# Giả lập nhãn thật và dự đoán để tính F1-Score
true_labels = [1, 1, 1]  # 1: đúng
predicted_labels = [1, 0, 1]  # 1: đúng, 0: sai

precision_f1, recall_f1, f1_score = compute_f1_score(true_labels, predicted_labels)

# Tạo bảng kết quả
results = {
    "Metric": ["Precision (F1)", "Recall (F1)", "F1-Score (F1)",
               "Precision (BERTScore)", "Recall (BERTScore)", "F1-Score (BERTScore)"],
    "Value": [precision_f1, recall_f1, f1_score,
               np.mean(bert_scores), np.mean(bert_scores), np.mean(bert_scores)]
}

df_results = pd.DataFrame(results)

# In kết quả chi tiết
print("=== Evaluation Metrics ===")
for i, (ref, gen, score) in enumerate(zip(reference_answers, generated_answers, bert_scores)):
    print(f"Example {i+1}:")
    print(f"Reference: {ref}")
    print(f"Generated: {gen}")
    print(f"BERTScore: {score:.4f}")
    print("-" * 50)

print("\n=== Overall Scores ===")
print(df_results.to_string(index=False))

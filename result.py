import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support
from bert_score import score

def cosine_similarity(a, b):
    """Tính độ tương đồng cos giữa hai vector"""
    a = a / np.linalg.norm(a)  # Chuẩn hóa vector a
    b = b / np.linalg.norm(b)  # Chuẩn hóa vector b
    return np.dot(a, b)

def compute_bertscore(reference, candidate, model, tokenizer):
    """Tính BERTScore giữa hai văn bản"""
    with torch.no_grad():
        ref_tokens = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
        cand_tokens = tokenizer(candidate, return_tensors="pt", padding=True, truncation=True)
        
        # Tính embedding cho câu tham chiếu và câu sinh ra
        ref_embedding = model(**ref_tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        cand_embedding = model(**cand_tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Tính Cosine Similarity
        score = cosine_similarity(ref_embedding, cand_embedding)
    return score

def compute_precision_recall_f1(reference, candidate, model, tokenizer):
    """Tính Precision, Recall, và F1 từ BERTScore"""
    # Tính BERTScore
    cosine = compute_bertscore(reference, candidate, model, tokenizer)

    # Để tính Precision, Recall, và F1, bạn cần sử dụng kết quả của BERTScore
    # BERTScore có thể chia thành precision, recall, và f1
    precision, recall, f1 = score(generated_answers, reference_answers, model_type=bert_model, verbose=True)

    return cosine ,precision, recall, f1

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

# Tính Precision, Recall và F1 cho mỗi cặp câu
results = []

for i, (ref, gen) in enumerate(zip(reference_answers, generated_answers)):
    cosine, precision, recall, f1 = compute_precision_recall_f1(ref, gen, model, tokenizer)
    results.append({
        "Số tt": i + 1,
        "Cosine Similarity": cosine,
        "Precision (BERTScore)": precision,
        "Recall (BERTScore)": recall,
        "F1-Score (BERTScore)": f1
    })

# Chuyển dữ liệu vào DataFrame
df_results = pd.DataFrame(results)

# In bảng kết quả
print("=== Bảng Kết Quả ===")
print(df_results.to_string(index=False))

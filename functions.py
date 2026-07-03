import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import code_bert_score

#model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def text_vectorize_score(prediction: str, g_truth: str, embed_model, bert_scorer):
    # Cosine Similarity

    pred_vector = embed_model.encode(prediction)
    truth_vector = embed_model.encode(g_truth)

    dot_product = np.dot(pred_vector, truth_vector)
    norm_pred = np.linalg.norm(pred_vector)
    norm_truth = np.linalg.norm(truth_vector)

    cos_similarity = dot_product / (norm_pred * norm_truth)

    # BERTScore

    pred_input = [prediction]
    truth_input = [g_truth]

    P, R, F1 = bert_scorer.score(pred_input, truth_input)

    P = P.item()
    R = R.item()
    F1 = F1.item()

    return {
        "cosine_similarity": cos_similarity,
        "BERTScore_F1": F1,
        "BERTScore_P": P,
        "BERTScore_R": R 
    }

tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model = AutoModel.from_pretrained("microsoft/unixcoder-base")

def code_vectorize_score(prediction: str, g_truth: str, tokenizer, model, bert_model):

    #Cosine Similarity for Code
    def get_code_embedding(code_snippet):
        tokens = tokenizer(
            f"<encoder-only> {code_snippet}",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**tokens)

        embeddings = outputs.last_hidden_state
        attention_mask = tokens['attention_mask'].unsqueeze(-1)

        masked_embeddings = embeddings * attention_mask
        sum_embeddings = torch.sum(masked_embeddings, dim = 1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask
    
    vector_pred = get_code_embedding(prediction)
    vector_truth = get_code_embedding(g_truth)

    similarity = F.cosine_similarity(vector_truth, vector_pred)

    similarity = similarity.item()

    #BERTScore for Code
    pred_input = [prediction]
    truth_input = [g_truth]

    P, R, F1, F3 = bert_model.score(
        cands = pred_input,
        refs = truth_input,
        model_type = "microsoft/unixcoder-base",
        lang = "python"
    )

    P1 = P.item()
    R1 = R.item()
    F1_1 = F1.item()

    return {
        "cosine_similarity": similarity,
        "BERTScore_F1": F1_1,
        "BERTScore_P": P1,
        "BERTScore_R": R1 
    }

def data_completion_score(prediction: float, g_truth: float):
    #Simple MSE

    return (g_truth - prediction) ** 2
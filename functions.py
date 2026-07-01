import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score

def text_vectorize_score(prediction: str, g_truth: str):
    # Cosine Similarity
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    pred_vector = model.encode(prediction)
    truth_vector = model.encode(g_truth)

    dot_product = np.dot(pred_vector, truth_vector)
    norm_pred = np.linalg.norm(pred_vector)
    norm_truth = np.linalg.norm(truth_vector)

    cos_similarity = dot_product / (norm_pred * norm_truth)

    # BERTScore

    P, R, F1 = score(prediction, g_truth, lang="en", verbose = False, model_type="distilbert-base-uncased")

    return {
        "cosine_similarity": cos_similarity,
        "BERTScore_F1": F1,
        "BERTScore_P": P,
        "BERTScore_R": R 
    }

def code_vectorize_score(prediction: str, g_truth: str):
    pass

def data_completion_score(prediction: float, g_truth: float):
    pass
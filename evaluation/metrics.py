import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import bert_score
import torch
import re

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')


def preprocess_text(text):
    """
    Preprocess text by lowercase, removing special characters, etc.
    
    Args:
        text (str): Input text.
        
    Returns:
        str: Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def tokenize_text(text):
    """
    Tokenize text into words.
    
    Args:
        text (str): Input text.
        
    Returns:
        list: List of tokens.
    """
    return nltk.word_tokenize(text)


def compute_bleu(references, hypotheses):
    """
    Compute BLEU score.
    
    Args:
        references (list): List of reference texts.
        hypotheses (list): List of hypothesis texts.
        
    Returns:
        dict: BLEU scores for different n-grams.
    """
    # Preprocess and tokenize texts
    references_tokenized = [tokenize_text(preprocess_text(ref)) for ref in references]
    hypotheses_tokenized = [tokenize_text(preprocess_text(hyp)) for hyp in hypotheses]
    
    # Initialize smoothing function
    smoothie = SmoothingFunction().method1
    
    # Compute BLEU scores for different n-grams
    bleu_1 = np.mean([
        sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        for ref, hyp in zip(references_tokenized, hypotheses_tokenized)
    ])
    
    bleu_2 = np.mean([
        sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        for ref, hyp in zip(references_tokenized, hypotheses_tokenized)
    ])
    
    bleu_3 = np.mean([
        sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        for ref, hyp in zip(references_tokenized, hypotheses_tokenized)
    ])
    
    bleu_4 = np.mean([
        sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        for ref, hyp in zip(references_tokenized, hypotheses_tokenized)
    ])
    
    return {
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_3': bleu_3,
        'bleu_4': bleu_4
    }


def compute_rouge(references, hypotheses):
    """
    Compute ROUGE score.
    
    Args:
        references (list): List of reference texts.
        hypotheses (list): List of hypothesis texts.
        
    Returns:
        dict: ROUGE scores.
    """
    # Initialize Rouge scorer
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    
    # Compute ROUGE scores
    scores = {
        'rouge_1_precision': 0.0,
        'rouge_1_recall': 0.0,
        'rouge_1_f1': 0.0,
        'rouge_2_precision': 0.0,
        'rouge_2_recall': 0.0,
        'rouge_2_f1': 0.0,
        'rouge_l_precision': 0.0,
        'rouge_l_recall': 0.0,
        'rouge_l_f1': 0.0
    }
    
    for ref, hyp in zip(references, hypotheses):
        # Preprocess text
        ref = preprocess_text(ref)
        hyp = preprocess_text(hyp)
        
        # If hypothesis is empty, skip
        if not hyp:
            continue
        
        # Compute scores
        result = scorer.score(ref, hyp)
        
        # Update scores
        scores['rouge_1_precision'] += result['rouge1'].precision
        scores['rouge_1_recall'] += result['rouge1'].recall
        scores['rouge_1_f1'] += result['rouge1'].fmeasure
        
        scores['rouge_2_precision'] += result['rouge2'].precision
        scores['rouge_2_recall'] += result['rouge2'].recall
        scores['rouge_2_f1'] += result['rouge2'].fmeasure
        
        scores['rouge_l_precision'] += result['rougeL'].precision
        scores['rouge_l_recall'] += result['rougeL'].recall
        scores['rouge_l_f1'] += result['rougeL'].fmeasure
    
    # Calculate average scores
    n = len(references)
    for key in scores:
        scores[key] /= n
    
    return scores


def compute_meteor(references, hypotheses):
    """
    Compute METEOR score.
    
    Args:
        references (list): List of reference texts.
        hypotheses (list): List of hypothesis texts.
        
    Returns:
        float: METEOR score.
    """
    # Preprocess and tokenize texts
    references_tokenized = [tokenize_text(preprocess_text(ref)) for ref in references]
    hypotheses_tokenized = [tokenize_text(preprocess_text(hyp)) for hyp in hypotheses]
    
    # Compute METEOR scores
    scores = [
        meteor_score([ref], hyp)
        for ref, hyp in zip(references_tokenized, hypotheses_tokenized)
    ]
    
    # Calculate average score
    meteor = np.mean(scores)
    
    return {'meteor': meteor}


def compute_bertscore(references, hypotheses):
    """
    Compute BERTScore.
    
    Args:
        references (list): List of reference texts.
        hypotheses (list): List of hypothesis texts.
        
    Returns:
        dict: BERTScore precision, recall, and F1.
    """
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Compute BERTScore
    P, R, F1 = bert_score.score(
        hypotheses,
        references,
        lang="en",
        rescale_with_baseline=True,
        device=device
    )
    
    # Calculate average scores
    precision = P.mean().item()
    recall = R.mean().item()
    f1 = F1.mean().item()
    
    return {
        'bertscore_precision': precision,
        'bertscore_recall': recall,
        'bertscore_f1': f1
    }


def compute_metrics(references, hypotheses, metrics=None):
    """
    Compute evaluation metrics.
    
    Args:
        references (list): List of reference texts.
        hypotheses (list): List of hypothesis texts.
        metrics (list, optional): List of metrics to compute.
            Options: 'bleu', 'rouge', 'meteor', 'bertscore'.
            If None, all metrics are computed.
            
    Returns:
        dict: Evaluation metrics.
    """
    if metrics is None:
        metrics = ['bleu', 'rouge', 'meteor', 'bertscore']
    
    results = {}
    
    # Compute metrics
    if 'bleu' in metrics:
        results.update(compute_bleu(references, hypotheses))
    
    if 'rouge' in metrics:
        results.update(compute_rouge(references, hypotheses))
    
    if 'meteor' in metrics:
        results.update(compute_meteor(references, hypotheses))
    
    if 'bertscore' in metrics:
        results.update(compute_bertscore(references, hypotheses))
    
    return results 
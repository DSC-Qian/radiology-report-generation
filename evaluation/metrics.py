"""
Evaluation metrics for radiology report generation.

This module provides metrics to evaluate the quality of generated radiology reports:
1. ROUGE-L for text overlap
2. BERTScore with biomedical models for semantic similarity
3. Radiological term detection for domain-specific accuracy
"""

import numpy as np
import torch
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import traceback
import re
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk

# Import the metrics libraries
try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    # Try to import the bert_score module to register custom models
    from bert_score import scorer as bert_scorer
except ImportError:
    pass  # We'll handle this in the respective functions

# Configure logging
logger = logging.getLogger(__name__)

# Common radiological findings and anatomical terms
RADIOLOGY_TERMS = [
    'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 
    'enlarged cardiomediastinum', 'fracture', 'lung lesion', 'lung opacity',
    'no finding', 'pleural effusion', 'pleural other', 'pneumonia', 
    'pneumothorax', 'support devices'
]

# Make sure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

def compute_rouge_l(references, hypotheses):
    """
    Compute ROUGE-L scores for a list of reference and hypothesis texts.
    
    Args:
        references (list): List of reference (ground truth) texts
        hypotheses (list): List of generated texts to evaluate
        
    Returns:
        dict: Dictionary with ROUGE-L precision, recall, and F1 scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.error("rouge_score package not found. Please install with: pip install rouge-score")
        return {"error": "rouge_score package not found"}
    
    logger.info("Computing ROUGE-L scores...")
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            logger.warning("Empty reference or hypothesis text, skipping")
            continue
        
        # Compute ROUGE-L score
        score = scorer.score(ref, hyp)
        scores.append(score['rougeL'])
    
    # Calculate averages
    if not scores:
        logger.error("No valid pairs for ROUGE-L calculation")
        return {
            "rougeL_precision": 0.0,
            "rougeL_recall": 0.0,
            "rougeL_f1": 0.0,
            "valid_pairs": 0,
            "total_pairs": len(references)
        }
    
    results = {
        "rougeL_precision": np.mean([s.precision for s in scores]),
        "rougeL_recall": np.mean([s.recall for s in scores]),
        "rougeL_f1": np.mean([s.fmeasure for s in scores]),
        "valid_pairs": len(scores),
        "total_pairs": len(references)
    }
    
    logger.info(f"ROUGE-L calculation complete. Average F1: {results['rougeL_f1']:.4f}")
    return results

def compute_bert_score(references, hypotheses, model_type="roberta-large"):
    """
    Compute BERTScore using a language model.
    
    Args:
        references (list): List of reference texts
        hypotheses (list): List of hypothesis texts
        model_type (str): Pretrained model to use for BERTScore
        
    Returns:
        dict: Dictionary with BERTScore precision, recall, and F1 scores
    """
    try:
        from bert_score import score
    except ImportError:
        logger.error("bert_score package not found. Please install with: pip install bert-score")
        return {"error": "bert_score package not found"}
    
    logger.info(f"Computing BERTScore with {model_type}...")
    
    # Filter out empty pairs
    valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) if ref and hyp]
    if not valid_pairs:
        logger.error("No valid pairs for BERTScore calculation")
        return {
            "bertScore_precision": 0.0,
            "bertScore_recall": 0.0, 
            "bertScore_f1": 0.0,
            "valid_pairs": 0,
            "total_pairs": len(references)
        }
    
    valid_refs, valid_hyps = zip(*valid_pairs)
    
    try:
        # Calculate BERTScore - using roberta-large which is known to work well
        P, R, F1 = score(valid_hyps, valid_refs, lang="en", model_type=model_type, verbose=True)
        
        results = {
            "bertScore_precision": P.mean().item(),
            "bertScore_recall": R.mean().item(),
            "bertScore_f1": F1.mean().item(),
            "valid_pairs": len(valid_pairs),
            "total_pairs": len(references)
        }
        
        logger.info(f"BERTScore calculation complete. Average F1: {results['bertScore_f1']:.4f}")
        return results
    
    except Exception as e:
        # If the model fails, log error and return empty metrics
        logger.error(f"Error computing BERTScore with {model_type}: {e}")
        return {
            "bertScore_precision": 0.0,
            "bertScore_recall": 0.0,
            "bertScore_f1": 0.0,
            "bertScore_error": str(e),
            "valid_pairs": 0,
            "total_pairs": len(references)
        }

def detect_radiological_terms(text):
    """
    Detect radiological terms in text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        set: Set of detected radiological terms
    """
    if not text:
        return set()
    
    text_lower = text.lower()
    found_terms = set()
    
    # Simple term matching
    for term in RADIOLOGY_TERMS:
        # Check if term exists but watch for word boundaries
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            found_terms.add(term)
    
    # Handle negations simply (could be much more sophisticated)
    negated_terms = set()
    for term in found_terms:
        if f"no {term}" in text_lower or f"without {term}" in text_lower:
            negated_terms.add(term)
    
    # Remove negated terms
    return found_terms - negated_terms

def compute_radiological_term_metrics(references, hypotheses):
    """
    Compute metrics based on radiological term detection.
    
    Args:
        references (list): List of reference texts
        hypotheses (list): List of hypothesis texts
        
    Returns:
        dict: Dictionary with precision, recall, and F1 scores
    """
    logger.info("Computing radiological term metrics...")
    
    # Process each pair
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    valid_pairs = 0
    
    for ref, hyp in tqdm(zip(references, hypotheses), total=len(references), desc="Analyzing medical terms"):
        if not ref or not hyp:
            continue
            
        ref_terms = detect_radiological_terms(ref)
        hyp_terms = detect_radiological_terms(hyp)
        
        # Calculate metrics
        tp = len(ref_terms.intersection(hyp_terms))
        fp = len(hyp_terms.difference(ref_terms))
        fn = len(ref_terms.difference(hyp_terms))
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        valid_pairs += 1
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        "radiology_precision": precision,
        "radiology_recall": recall,
        "radiology_f1": f1,
        "valid_pairs": valid_pairs,
        "total_pairs": len(references)
    }
    
    logger.info(f"Radiological term metric calculation complete. F1: {f1:.4f}")
    return results

def compute_chexbert_metrics(references, hypotheses):
    """
    Compute CheXbert-like metrics for clinical correctness evaluation.
    
    This is a simplified implementation of CheXbert labeling that uses the radiological terms
    detection as a baseline and extends it with more sophisticated rules for negation,
    uncertainty, and more accurate term matching. In a production system, this would be
    replaced with the actual CheXbert model.
    
    Args:
        references (list): List of reference texts
        hypotheses (list): List of hypothesis texts
        
    Returns:
        dict: Dictionary with precision, recall, and F1 scores for clinical findings
    """
    logger.info("Computing CheXbert-like clinical metrics...")
    
    # CheXpert labels (based on the CheXbert paper)
    CHEXPERT_LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
        'Pneumothorax', 'Support Devices'
    ]
    
    # Label values (0: Negative, 1: Positive, -1: Uncertain, 2: Not Mentioned)
    LABEL_VALUES = {
        'negative': 0,
        'positive': 1,
        'uncertain': -1,
        'not_mentioned': 2
    }
    
    # Keywords for uncertainty
    UNCERTAINTY_TERMS = [
        'possible', 'possibly', 'probable', 'probably', 'likely', 'may', 'might',
        'can', 'could', 'suspicious', 'suspicion', 'suggest', 'appears', 'apparent',
        'compatible with'
    ]
    
    # Enhanced negation terms
    NEGATION_TERMS = [
        'no', 'not', 'without', 'free of', 'clear of', 'negative for', 'absence of',
        'resolved', 'no evidence of', 'no sign of', 'no indication of'
    ]
    
    def extract_chexpert_labels(text):
        """Helper function to extract CheXpert-like labels from text."""
        if not text:
            return {label: LABEL_VALUES['not_mentioned'] for label in CHEXPERT_LABELS}
        
        text_lower = text.lower()
        results = {}
        
        for label in CHEXPERT_LABELS:
            label_lower = label.lower()
            
            # Not mentioned by default
            results[label] = LABEL_VALUES['not_mentioned']
            
            # Basic term matching (case insensitive)
            match = re.search(r'\b' + re.escape(label_lower) + r'\b', text_lower)
            if not match:
                continue
            
            # Found a mention, assume positive initially
            results[label] = LABEL_VALUES['positive']
            
            # Check for negation (within reasonable distance from the term)
            context_start = max(0, match.start() - 50)
            context_end = min(len(text_lower), match.end() + 50)
            context = text_lower[context_start:context_end]
            
            for neg_term in NEGATION_TERMS:
                if neg_term in context and context.find(neg_term) < context.find(label_lower):
                    results[label] = LABEL_VALUES['negative']
                    break
            
            # Check for uncertainty (if not already negated)
            if results[label] == LABEL_VALUES['positive']:
                for uncertain_term in UNCERTAINTY_TERMS:
                    if uncertain_term in context and abs(context.find(uncertain_term) - context.find(label_lower)) < 30:
                        results[label] = LABEL_VALUES['uncertain']
                        break
        
        # Special case: No Finding
        if 'No Finding' in CHEXPERT_LABELS:
            # If no positive findings are detected, then "No Finding" is positive
            if not any(v == LABEL_VALUES['positive'] for k, v in results.items() if k != 'No Finding'):
                if 'normal' in text_lower or 'no abnormality' in text_lower or 'unremarkable' in text_lower:
                    results['No Finding'] = LABEL_VALUES['positive']
        
        return results
    
    # Process each reference-hypothesis pair
    pred_labels = []
    true_labels = []
    label_counts = {label: 0 for label in CHEXPERT_LABELS}
    
    for ref, hyp in tqdm(zip(references, hypotheses), total=len(references), desc="Analyzing clinical findings"):
        if not ref or not hyp:
            continue
        
        ref_labels = extract_chexpert_labels(ref)
        hyp_labels = extract_chexpert_labels(hyp)
        
        # Collect positive/negative/uncertain mentions for each label (excluding not_mentioned)
        for label in CHEXPERT_LABELS:
            if ref_labels[label] != LABEL_VALUES['not_mentioned'] or hyp_labels[label] != LABEL_VALUES['not_mentioned']:
                true_labels.append((label, ref_labels[label]))
                pred_labels.append((label, hyp_labels[label]))
                label_counts[label] += 1
    
    # Calculate agreement, precision, recall, F1
    correct = 0
    total = len(true_labels)
    
    # Label-based metrics for each finding
    label_metrics = {}
    
    for label in CHEXPERT_LABELS:
        label_true = [t[1] for t in true_labels if t[0] == label]
        label_pred = [p[1] for p in pred_labels if p[0] == label]
        
        # Skip if no instances of this label
        if not label_true:
            continue
        
        # Count exact matches
        label_correct = sum(1 for t, p in zip(label_true, label_pred) if t == p)
        
        # Calculate precision and recall
        label_metrics[label] = {
            'count': len(label_true),
            'accuracy': label_correct / len(label_true) if label_true else 0
        }
        
        # Add to overall correct count
        correct += label_correct
    
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Prepare results
    results = {
        "chexbert_accuracy": accuracy,
        "total_findings": total,
        "finding_counts": label_counts
    }
    
    # Add individual label metrics
    for label, metrics in label_metrics.items():
        results[f"chexbert_{label.lower().replace(' ', '_')}_accuracy"] = metrics['accuracy']
    
    logger.info(f"CheXbert-like clinical metric calculation complete. Accuracy: {accuracy:.4f}")
    return results

def compute_metrics(references, hypotheses, device=None):
    """
    Compute all metrics for the generated reports.
    
    Args:
        references (list): List of reference texts
        hypotheses (list): List of hypothesis texts
        device: Ignored, kept for compatibility
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Check input validity
    if not references or not hypotheses or len(references) != len(hypotheses):
        logger.error("Invalid input: references or hypotheses list is empty or lengths do not match.")
        return {"error": "Invalid input for metric computation."}

    all_metrics = {}
    
    # Compute ROUGE-L
    try:
        rouge_metrics = compute_rouge_l(references, hypotheses)
        all_metrics.update(rouge_metrics)
    except Exception as e:
        logger.error(f"Error computing ROUGE metrics: {e}")
        all_metrics.update({"rouge_error": str(e)})
    
    # Compute BERTScore
    try:
        bert_metrics = compute_bert_score(references, hypotheses)
        all_metrics.update(bert_metrics)
    except Exception as e:
        logger.error(f"Error computing BERTScore metrics: {e}")
        all_metrics.update({"bert_score_error": str(e)})
    
    # Compute radiological term metrics
    try:
        rad_metrics = compute_radiological_term_metrics(references, hypotheses)
        all_metrics.update(rad_metrics)
    except Exception as e:
        logger.error(f"Error computing radiological term metrics: {e}")
        all_metrics.update({"rad_metrics_error": str(e)})
    
    # Compute CheXbert-like metrics
    try:
        chexbert_metrics = compute_chexbert_metrics(references, hypotheses)
        all_metrics.update(chexbert_metrics)
    except Exception as e:
        logger.error(f"Error computing CheXbert metrics: {e}")
        all_metrics.update({"chexbert_error": str(e)})
    
    return all_metrics

# Example Usage
if __name__ == '__main__':
    # This block will only run if the script is executed directly
    print("Running example usage of radiology report metrics...")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example reference and hypothesis reports
    example_references = [
        "Findings: Cardiomegaly is present. No pleural effusion. Lungs are clear.",
        "Impression: Mild pulmonary edema. No pneumothorax.",
        "There is consolidation in the left lower lobe consistent with pneumonia.",
        "No acute cardiopulmonary abnormality.", # Example: No Finding
        "Normal chest radiograph with no evidence of active disease.",
        "Support device noted." # Example: Matches Support Device
    ]
    example_hypotheses = [
        "Cardiomegaly noted. Lungs clear.", # Matches Cardiomegaly, misses effusion negation
        "Mild edema. No pneumothorax seen.", # Matches Edema and pneumothorax negation
        "Findings suggest pneumonia in the left lung base.", # Matches pneumonia
        "Normal chest x-ray.", # Matches No Finding (implicitly)
        "The chest radiograph shows no abnormalities.", # Example: No findings
        "Central line is present." # Example: Matches Support Device
    ]

    print("\nExample References:")
    for r in example_references: print(f"- '{r}'")
    print("\nExample Hypotheses:")
    for h in example_hypotheses: print(f"- '{h}'")

    try:
        # Compute metrics
        metrics_results = compute_metrics(example_references, example_hypotheses)

        print("\nComputed Metrics:")
        if metrics_results:
            if 'error' in metrics_results:
                 print(f"Metric computation failed: {metrics_results['error']}")
            else:
                for name, value in metrics_results.items():
                    # Format floats, print others directly
                    if isinstance(value, (float, np.float_, np.float32, np.float64)): 
                         print(f"- {name}: {value:.4f}")
                    else:
                         print(f"- {name}: {value}")
        else:
            print("Metric computation returned empty with no error specified.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during example execution: {e}")
        print(traceback.format_exc())

    print("\nExample usage finished.") 
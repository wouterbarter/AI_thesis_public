import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

def compute_entropy(probs, epsilon=1e-9):
    """Computes Shannon entropy for a probability distribution."""
    # Use numpy or torch depending on input
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()
    
    # Clamp for stability
    probs = np.clip(probs, epsilon, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def calculate_irr_metrics(h1, h2, llm, metric='kappa'):
    """
    Calculates the primary association metric (Kappa or Spearman's rho).
    Returns: (H-H association, H1-LLM association, H2-LLM association)
    """
    if metric == 'kappa':
        # Cohen's Kappa (Quadratic Weighted) - Agreement Metric
        k_hh = cohen_kappa_score(h1, h2, weights='quadratic')
        k_h1_m = cohen_kappa_score(h1, llm, weights='quadratic')
        k_h2_m = cohen_kappa_score(h2, llm, weights='quadratic')
        
    elif metric == 'spearman':
        # Spearman's Rho (Rank Correlation) - Association Metric
        # [0] = correlation coefficient, [1] = p-value
        k_hh = spearmanr(h1, h2)[0]
        k_h1_m = spearmanr(h1, llm)[0]
        k_h2_m = spearmanr(h2, llm)[0]
        
    else:
        raise ValueError("Metric must be 'kappa' or 'spearman'.")
        
    return k_hh, k_h1_m, k_h2_m

def calculate_reliability_gap(h1, h2, llm, metric='kappa'):
    """Calculates the Reliability Gap based on the chosen metric."""
    k_hh, k_h1_m, k_h2_m = calculate_irr_metrics(h1, h2, llm, metric=metric)
    
    # The gap logic remains the same regardless of whether k is Kappa or Rho
    k_llm_avg = (k_h1_m + k_h2_m) / 2
    gap = k_hh - k_llm_avg
    
    return gap, k_hh, k_llm_avg
















def compute_validity_mass(top_tokens, top_logits, valid_set={'1','2','3','4'}):
    """Checks for probability leakage."""
    # (Insert the logic we wrote previously for validity mass)
    pass
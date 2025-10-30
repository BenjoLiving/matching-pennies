import numpy as np 

def log_odds_to_prob(log_odds): 
    odds = np.exp(log_odds) 
    p = (odds) / (1 + odds)
    return p
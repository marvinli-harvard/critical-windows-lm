
import numpy as np
import pandas as pd
from utils.utils import * 
from typing import Tuple

def compute_ci(data : pd.Series, num_samples:int=1000, ci:int=95) -> Tuple[float, float, float]:

    """
    Compute the confidence interval for the mean of a given dataset using bootstrapping.
    Args:
        data (pd.Series): The input data series for which the confidence interval is to be computed.
        num_samples (int, optional): The number of bootstrap samples to generate. Default is 1000.
        ci (int, optional): The confidence level for the interval. Default is 95.
    Returns:
        Tuple[float, float, float]: A tuple containing the mean of the data, the lower bound of the confidence interval, and the upper bound of the confidence interval.
    """
    
    resampled_means = np.random.choice(data, (num_samples, len(data))).mean(axis=1)
    lower_bound = np.percentile(resampled_means, (100 - ci) / 2)
    upper_bound = np.percentile(resampled_means, 100 - (100 - ci) / 2)
    return data.mean(), lower_bound, upper_bound


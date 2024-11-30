from typing import Tuple, Optional

import numpy as np
import pandas as pd 

from utils.utils import * 

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

def display_then_plot(combined_df : pd.DataFrame, 
                      column : str, 
                      base_column : Optional[str],
                      label : Optional[str]
                     ) -> None :
    curr_df = (combined_df
            .groupby(["dataset_source", "stop_frac"])[[column]].mean().reset_index()
             .pivot(index="dataset_source", columns="stop_frac", values=column)
    )
    if base_column:
        curr_df = curr_df.join(
            combined_df[["dataset_source","problem",base_column]]
            .drop_duplicates()
            .groupby("dataset_source")[base_column]
            .mean()
        )
    
    curr_df.T.plot(xlabel="% of prompt remaining",ylabel=label)
    return curr_df 

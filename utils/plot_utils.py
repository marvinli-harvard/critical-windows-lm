from typing import Tuple, Optional

import math
import numpy as np
import pandas as pd
import torch 
import matplotlib.pyplot as plt 
from scipy.stats import bootstrap
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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

def compute_tprs(train_statistics, val_statistics, fprs=None):
    '''
    Compute TPRS at FPRS with train and validation test statistics. 
    
    **Note that we assume train statistic < test statistic. Negate before using if otherwise.**

    Args:
        train_statistics (list[float]): list of train statistics
        val_statistics (list[float]): list of val statistics
        plot_title (str): title of the plot
        ci (bool): compute confidence intervals. Default: True
        num_bootstraps (int): number of bootstraps for confidence interval
        keep_first (int): compute only for the first keep_first number of samples
        show_plot (bool): whether to show the plot
        save_name (str): save path for plot and scores (without extension); does not save unless save_name is specified
        log_scale (bool): whether to plot on log-log scale
        lims (list): argument to xlim and ylim
        fprs (list[float]): return TPRs at given FPRs. If unspecified, calculates at every 0.1 increment
        color (str): color
    
    Returns:
        auc (float): the ROC-AUC score
        tpr_at_fprs (list[float]): the tprs at the given fprs
    '''
    # Preprocess
    train_statistics = torch.as_tensor(train_statistics).flatten()
    train_statistics = train_statistics[~train_statistics.isnan()]
    val_statistics = torch.as_tensor(val_statistics).flatten()
    val_statistics = val_statistics[~val_statistics.isnan()]

    ground_truth = torch.cat((torch.ones_like(train_statistics),torch.zeros_like(val_statistics))).flatten()
    predictions = torch.cat((-train_statistics,-val_statistics)).flatten()
    n_points = len(ground_truth)

    fpr, tpr, thresholds = roc_curve(ground_truth,predictions)

    # Process FPRs
    if fprs is None:
        fprs = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    tpr_at_fprs = [tpr[np.max(np.argwhere(fpr<=fpr_val))] for fpr_val in fprs]
    
    return pd.DataFrame({"FPRs":fprs,"TPRs":tpr_at_fprs})

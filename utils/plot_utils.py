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

def plot_ROC(train_statistics, val_statistics, plot_title, keep_first=None, ci=True, num_bootstraps=1000, fprs=None, log_scale=False, show_plot=True, save_name=None, lims=None, color='darkorange'):
    '''
    Plots ROC curve with train and validation test statistics. Also saves TPRs at FPRs.
    
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
    train_statistics = torch.as_tensor(train_statistics).flatten()[:keep_first]
    train_statistics = train_statistics[~train_statistics.isnan()]
    val_statistics = torch.as_tensor(val_statistics).flatten()[:keep_first]
    val_statistics = val_statistics[~val_statistics.isnan()]

    ground_truth = torch.cat((torch.ones_like(train_statistics),torch.zeros_like(val_statistics))).flatten()
    predictions = torch.cat((-train_statistics,-val_statistics)).flatten()
    n_points = len(ground_truth)

    fpr, tpr, thresholds = roc_curve(ground_truth,predictions)
    roc_auc = auc(fpr, tpr)

    # Process FPRs
    if fprs is None:
        fprs = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    tpr_at_fprs = [tpr[np.max(np.argwhere(fpr<=fpr_val))] for fpr_val in fprs]
    
    # Compute CI
    if ci:
        fpr_range = np.linspace(0, 1, n_points)
        def auc_statistic(data,axis):
            ground_truth = data[0,0,:].T
            predictions = data[1,0,:].T
            fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
            roc_auc = auc(fpr, tpr)
            tpr_range = np.interp(fpr_range,fpr,tpr)
            return np.array([[roc_auc]+tpr_range.tolist()]).T
        
        data = torch.cat((ground_truth[:,None],predictions[:,None]),dim=1)
        bootstrap_result = bootstrap((data,), auc_statistic, confidence_level=0.95, n_resamples=num_bootstraps, batch=1, method='percentile',axis=0)
        auc_se = bootstrap_result.standard_error[0]
        tpr_se = [bootstrap_result.standard_error[np.max(np.argwhere(fpr_range<=fpr_val))] for fpr_val in fprs]

    # Plot
    plt.figure(figsize=(7,7),dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", c="k")
    if not log_scale:
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:0.4f}',c=color)
        plt.xlim([0,1] if lims is None else lims)
        plt.ylim([0,1] if lims is None else lims)
    else:
        plt.loglog(fpr, tpr, label=f'AUC = {roc_auc:0.4f}',c=color)
        plt.xlim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
        plt.ylim([10**(-int(np.log10(n_points))),1] if lims is None else lims)
    if ci:
        plt.fill_between(fpr_range,bootstrap_result.confidence_interval.low[1:],bootstrap_result.confidence_interval.high[1:],alpha=0.1,color=color)
    plt.title(plot_title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.minorticks_on()
    plt.grid(which="major",alpha=0.2)
    plt.grid(which="minor",alpha=0.1)
    return pd.DataFrame({"FPRs":fprs,"TPRs":tpr_at_fprs})

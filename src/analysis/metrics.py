import os, math, pandas as pd, numpy as np, itertools, matplotlib
from scipy import stats
from matplotlib import pyplot as plt, cm
from matplotlib.colors import ListedColormap

def mae(ytrue, ypred, mean=True):
    maes = np.abs(ytrue - ypred).mean(axis=0)
    if mean: return maes.mean()
    return maes

def smape(ytrue, ypred, mean=True):
    smapes = 200 * (np.abs(ytrue - ypred) / (0.001 + np.abs(ytrue) + np.abs(ypred))).mean(axis=0)
    if mean: return smapes.mean()
    return smapes

def CC(ytrue, ypred, mean=True):
    ypred_centered  = ypred - ypred.mean(axis=0)
    ytrue_centered = ytrue - ytrue.mean(axis=0)

    num = (ypred_centered * ytrue_centered).sum(axis=0)
    d1 = (ypred_centered * ypred_centered).sum(axis=0)
    d2 = (ytrue_centered * ytrue_centered).sum(axis=0)
    d = np.sqrt(d1 * d2) + 0.001
    accs = num / d
    if mean:
        accs = accs.mean()
    return accs

def DM(ytrue, ypred_1, ypred_2, norm="smape"):
    e1 = ytrue - ypred_1
    e2 = ytrue - ypred_2

    # Computing the loss differential series for the multivariate test
    if norm == "mae":
        d = np.mean(np.abs(e1), axis=1) - np.mean(np.abs(e2), axis=1)
    if norm == "smape":
        d1=(200*np.abs(e1)/ (np.abs(ytrue) + np.abs(ypred_1) + 0.0001)).mean(axis=1)
        d2=(200*np.abs(e2)/(np.abs(ytrue) + np.abs(ypred_2) + 0.0001)).mean(axis=1)
        d = d1 - d2

    # Computing the test statistic
    mu = d.mean()
    sigma = d.var()
    DMs = mu / np.sqrt(sigma / d.size)

    # Compute the pvalue
    p_value = 1 - stats.norm.cdf(DMs)
    return p_value

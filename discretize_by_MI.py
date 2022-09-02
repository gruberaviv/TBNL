import pandas as pd
import numpy as np
from python_scripts.marg_prob import margprob
from python_scripts.entropy_from_counts import entropy_from_counts, cond_entropy_from_counts


def discretize(series, bins=3, range_or_quantile='range'):
    if range_or_quantile == 'range':
        return pd.cut(series, bins)
    if range_or_quantile == 'quantile':
        return pd.qcut(series, bins, duplicates='drop')
    raise Exception("Something is broken with the discretization input!!")


def try_discretizing(series, bins=3):
    try:
        return discretize(series, bins=bins, range_or_quantile='quantile')
    except Exception:
        try:
            return discretize(series, bins=bins)
        except Exception:
            print("Can't discretize", series.name)
            return series


def discretize_by_mi(continuous_X, target_variable, max_values=0, MC=0):
    # % discription:
    # % link:
    # %
    # %INPUT:
    # % continuous_X - original data values (only the column to be discretized)
    # % target_variable - target variable (column), same size as continuous_X
    # % max_values - maximal number of discretized values
    # %
    # % OUTPUT:
    # % discrete_X - column of discretized values, same size and same order as continuous_X

    # EXAMPLES
    # d <- discretize_by_mi(continuous_X = iris[,1], target_variable = iris[,5])
    # d <- discretize_by_mi(continuous_X = iris[,1], target_variable = iris[,5], max_values = 8) - will first discretize continuous_X to 8 equi-range bins
    # d <- discretize_by_mi(continuous_X = iris[,1], target_variable = iris[,5], MC = 1000) - will calculate the mutual information via monte carlo estimation

    D = pd.concat([continuous_X.reset_index(drop=True), target_variable.reset_index(drop=True)], axis=1).reset_index(
        drop=True)

    continuous_col = D.columns[0]
    target_col = D.columns[1]

    D_sorted = D.sort_values(continuous_col)

    nRows = D_sorted.shape[0]

    if max_values > 0:
        low_res_D_sorted = try_discretizing(D_sorted[continuous_col], bins=max_values)

        D_sorted[continuous_col] = low_res_D_sorted.apply(lambda x: x.right)

    JPD_vals, JPD_probs = margprob(D_sorted, MC=MC)

    Y_vals, Y_probs = margprob(JPD_vals, JPD_probs, target_col, MC=MC)

    # [Y_vals Y_probs] = MargProb(JPD_vals, JPD_probs, [0 1]);
    Hy = entropy_from_counts(Y_probs)
    # Hy = ConditionalEntropy(Y_vals, Y_probs);
    cond_H = cond_entropy_from_counts(JPD_vals, JPD_probs, D.columns[1])
    # cond_H = ConditionalEntropy(JPD_vals, JPD_probs, [0 1]);
    joint_H = entropy_from_counts(JPD_probs)
    # joint_H = ConditionalEntropy(JPD_vals, JPD_probs);
    I_old = Hy - cond_H
    R_old = I_old / joint_H
    halamish = D_sorted.copy().reset_index()
    j = 0
    while j >= 0:
        j = -1
        j_old = j
        for i in range(nRows - 1):
            if halamish.loc[i][continuous_col] < halamish.loc[i + 1][continuous_col]:
                old_halamish = halamish.loc[i][continuous_col]
                halamish.loc[i, continuous_col] = halamish.loc[i + 1][continuous_col]
                k = i - 1
                if k >= 0:
                    while halamish.loc[k][continuous_col] == old_halamish:
                        halamish.loc[k, continuous_col] = halamish.loc[i + 1][continuous_col]
                        k = k - 1
                        if k < 0:
                            break
                temp_JPD_vals, temp_JPD_probs = margprob(halamish[[continuous_col, target_col]], MC=MC)
                cond_H = cond_entropy_from_counts(temp_JPD_vals, temp_JPD_probs, target_col)
                joint_H = entropy_from_counts(temp_JPD_probs)
                I_new = Hy - cond_H
                R_new = I_new / joint_H
                if R_new > R_old:
                    R_old = R_new
                    j = i

                halamish.loc[(k + 1):i, continuous_col] = old_halamish

            if (j >= 0) & (j != j_old):
                j_old = j
                old_halamish = halamish.loc[j][continuous_col]
                halamish.loc[j, continuous_col] = halamish.loc[j + 1][continuous_col]
                k = j - 1
                if k >= 0:
                    while halamish.loc[k][continuous_col] == old_halamish:
                        halamish.loc[k, continuous_col] = halamish.loc[j + 1][continuous_col]
                        k = k - 1
                        if k < 0:
                            break

    # halamish = pd.concat([halamish, d2])
    # q2 <- order(halamish[,3])
    # sofsof <- halamish[q2,]
    # discrete_X = sofsof[,1]

    discrete_X = halamish.sort_values('index')[continuous_col]

    original_vals = continuous_X.nunique()
    upper_bounds = discrete_X.unique()
    # upper_bounds = upper_bounds[order(upper_bounds)]

    discretized_vals = len(upper_bounds)

    print('Reduced from', original_vals, 'to', discretized_vals)
    print(upper_bounds)

    return np.array(discrete_X)

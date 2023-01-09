
import itertools
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import copy
import math





def get_uniq_vals_in_arr(arr):
    uniq_vals = []
    for id_col in range(arr.shape[1]):
        uniq_vals.append(np.unique(arr[:, id_col]).tolist())
    
    return uniq_vals


def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def get_info_coef(left, right):
    # Both arrays NEED same number of rows
    assert left.shape[0] == right.shape[0]
    num_rows = left.shape[0]
    num_left_cols = left.shape[1]
        
    concat_mat = np.concatenate((left, right), axis=1)
    concat_uniq_vals = get_uniq_vals_in_arr(concat_mat)
    concat_combos = list(itertools.product(*concat_uniq_vals))
    p_sum = 0
    for vec in concat_combos:
        p_r1_r2 = len(np.where((concat_mat == vec).all(axis=1))[0]) / num_rows
        p_r1 = len(np.where((left == vec[:num_left_cols]).all(axis=1))[0]) / num_rows
        p_r2 = len(np.where((right == vec[num_left_cols:]).all(axis=1))[0]) / num_rows
        
        if p_r1_r2 == 0 or p_r1 == 0 or p_r2 == 0:
            p_iter = 0
        else:
            p_iter = p_r1_r2 * np.log(p_r1_r2 / p_r1) / p_r1
        p_sum += np.abs(p_iter)
    return p_sum

def get_conditional_info_coef(left, right, conditional): 
    assert (left.shape[0] == right.shape[0]) and (left.shape[0] == conditional.shape[0])
    num_rows = left.shape[0]
    num_left_cols = left.shape[1]
    num_right_cols = right.shape[1]

    right_concat_mat = np.concatenate((right, conditional), axis=1)    
    concat_mat = np.concatenate((left, right_concat_mat), axis=1)
    concat_uniq_vals = get_uniq_vals_in_arr(concat_mat)
    concat_combos = list(itertools.product(*concat_uniq_vals))
    p_sum = 0
    for vec in concat_combos:
        p_r1_r2 = len(np.where((concat_mat == vec).all(axis=1))[0]) / num_rows
        p_r1 = len(np.where((left == vec[:num_left_cols]).all(axis=1))[0]) / num_rows
        p_r2 = len(np.where((concat_mat[:, num_left_cols: -num_right_cols] == vec[num_left_cols: -num_right_cols]).all(axis=1))[0]) / num_rows
        
        try:
            p_r1_given_r3 = len(np.where((concat_mat[:, :num_left_cols] == vec[:num_left_cols]).all(axis=1) & (concat_mat[:, -num_right_cols:] == vec[-num_right_cols:]).all(axis=1))[0]) / len(np.where((concat_mat[:, -num_right_cols:] == vec[-num_right_cols:]).all(axis=1))[0])
        except ZeroDivisionError:
            p_r1_given_r3 = 0
        
        if p_r1_r2 == 0 or p_r1 == 0 or p_r2 == 0 or p_r1_given_r3 == 0:
            p_iter = 0
        else:
            p_iter = p_r1_r2 * np.log(p_r1_r2 / p_r2) / p_r1_given_r3
        p_sum += np.abs(p_iter)
    return p_sum

def get_disc_coef(y, x_s, protected_attr):
    x_s_a = np.concatenate((x_s, protected_attr), axis=1)
    return get_info_coef(y, x_s_a) * get_info_coef(x_s, protected_attr) * get_conditional_info_coef(x_s, protected_attr, y)

def get_shapley_disc_i(y, x, protected_attr, i):
    num_features = x.shape[1]
    lst_idx = list(range(num_features))
    lst_idx.pop(i)
    power_set = [x for x in powerset(lst_idx) if len(x) > 0]
    
    shapley = 0
    for set_idx in power_set:
        coef = math.factorial(len(set_idx)) * math.factorial(num_features - len(set_idx) - 1) / math.factorial(num_features)
        
        # Calculate v_D(T U {i})
        idx_xs_incl = copy.copy(set_idx)
        idx_xs_incl.append(i)
        disc_incl = get_disc_coef(y.reshape(-1, 1), x[:, idx_xs_incl], protected_attr.reshape(-1, 1))
        
        # Calculate v_D(T)
        disc_excl = get_disc_coef(y.reshape(-1, 1), x[:, set_idx], protected_attr.reshape(-1, 1))
        
        marginal = disc_incl - disc_excl
        shapley = shapley + coef * marginal
    return shapley
# %%

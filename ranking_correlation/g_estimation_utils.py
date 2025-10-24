import gc
from itertools import permutations
import json
import math
import numpy as np
from params import *

from ranking_coefficients import n_to_x, n_to_x_log, poly_estimator, x_is_log, w_calculator, gamma_calc, f


# # READ pars_dic_dic
# with open('estimation_results/pars_dic_dic.json', 'r', encoding ='utf8') as json_file:
#     pars_dic_dic = json.load(json_file)


###########################################
####
# # coefficients optimized for g calculation

def w_calculator_permutations(bb, n_0, aa_w, weight_type=1, weighting_scheme='add'):
    bb_w = np.array([f(x, weight_type=weight_type, n_0=n_0) for x in bb])
    if weighting_scheme == 'add':
        return aa_w + bb_w
    elif weighting_scheme == 'mult':
        return aa_w * bb_w
    else:
        raise ValueError(f"Unexpected weighting_scheme: {weighting_scheme}")


def spearman_permutations(bb, w):
    n = len(bb)
    wsum = sum(w)
    aam = sum([w[i] * (i + 1) for i in range(n)]) / wsum
    bbm = sum([w[i] * bb[i] for i in range(n)]) / wsum
    ssa = math.sqrt(sum([w[i] * ((i + 1) ** 2 - aam ** 2) for i in range(n)]) / wsum)
    ssb = math.sqrt(sum([w[i] * (bb[i] ** 2 - bbm ** 2) for i in range(n)]) / wsum)
    coeff = sum([w[i] * (i + 1 - aam) * (bb[i] - bbm) for i in range(n)]) / (wsum * ssa * ssb)
    return coeff


def kendall_permutations(bb, w):
    n = len(bb)
    num = 0
    den = 0
    for i in range(0, n):
        for j in range(0, i):
            x = w[i] * w[j]
            num +=  - x * np.sign((bb[j] - bb[i]))
            den += x
        for j in range(i + 1, n):
            x = w[i] * w[j]
            num += x * np.sign((bb[j] - bb[i]))
            den += x
    return num / den


def gamma_permutations(bb, w, coeff_type):
    if coeff_type == 'spearman':
        return spearman_permutations(bb=bb, w=w)
    elif coeff_type == 'kendall':
        return kendall_permutations(bb=bb, w=w)
    else:
        raise ValueError(f'Unknown coeff_type: {coeff_type}') 

####
###########################################


def regressor(y_dic, deg, is_log, y_err_dic=None, is_train=False):
    n_list = list(y_dic.keys())
    if is_log:
        traformed_n_list = [n_to_x_log(n) for n in n_list]
    else:
        traformed_n_list = [n_to_x(n) for n in n_list]    
    y_list = list(y_dic.values())    
    if y_err_dic:
        weight_list = np.array([1 / y_err_dic[n]  for n in n_list])
    else:
        weight_list = np.ones(len(n_list))
    if is_train:
        traformed_n_list = traformed_n_list[::2]
        y_list = y_list[::2]
        weight_list = weight_list[::2]
    pars = list(np.poly1d(np.polyfit(x=traformed_n_list, y=y_list, deg=deg, w=weight_list)))[::-1]    
    return pars


def mse(y_dic, pars, is_log):
    n_list = list(y_dic.keys())
    return sum([(y_dic[n] - poly_estimator(n, pars, is_log)) ** 2 for n in n_list]) / len(n_list)


def bar_calculator(gamma_list, pars_dic=None, n=None):
    n_samp = len(gamma_list)
    
    # round 2: alpha_1 estimation
    if  pars_dic:
        if n > n_max_exact:
            is_log = pars_dic['is_log']
            pars = pars_dic['pars']
            gamma_bar = poly_estimator(n, pars, is_log=is_log)
        else:
            gamma_bar = np.mean(gamma_list)
        V = np.mean([(x - gamma_bar) ** 2 for x in gamma_list])        
        V_left = np.sum([(x - gamma_bar) ** 2 for x in gamma_list if x < gamma_bar]) / n_samp

        return '', '', V, V_left
    # round 1: gamma_bar estimation
    else:
        gamma_bar = np.mean(gamma_list)
        gamma_bar_sd = math.sqrt(np.var(gamma_list) / n_samp)
        return gamma_bar, gamma_bar_sd, '', ''


def n_samp_calc(coeff_type, n):
    n = int(n)
    if coeff_type == 'kendall':
        n_samp = n_samp_kendall
        if n > 1500:
            n_samp = int(round(n_samp / 5))
    elif coeff_type == 'spearman':
        n_samp = n_samp_spearman
    if n <= 100:
        n_samp = n_samp * 10
    return n_samp


def n_list_calc(coeff_type):
    if coeff_type == 'kendall':
        a_max = a_max_kendall
    elif coeff_type == 'spearman':
        a_max = a_max_spearman
    return [round(q ** a) for a in range(a_min, a_max + 1)]

    
def sample_looper(n_0, coeff_type, weight_type, weighting_scheme, n_list, is_round_1=True, pars_dic_dic=None):
    gamma_dic = {}
    V_dic = {}
    V_left_dic = {}
    err_dic = {}
    for n in n_list:
        n_samp = n_samp_calc(coeff_type=coeff_type, n=n)

        print(f'\r\tCalculating n = {n}\t', end='')
        aa = range(1, n + 1)
        aa_w = np.array([f(i, weight_type=weight_type, n_0=n_0) for i in aa])

        gamma_list_n = []
        for i in range(n_samp):
            bb = np.random.permutation(aa)
            w = w_calculator_permutations(bb=bb, n_0=n_0, aa_w = aa_w, weight_type=weight_type, weighting_scheme=weighting_scheme)
            gamma_list_n.append(gamma_permutations(bb=bb, w=w, coeff_type=coeff_type))
        if is_round_1:
            gamma_bar, gamma_bar_sd, _,_  = bar_calculator(gamma_list_n)
            gamma_dic[n] = gamma_bar
            err_dic[n] = gamma_bar_sd
        else:
            pars_dic=pars_dic_dic['gamma'][coeff_type][weighting_scheme][weight_type][n_0]

            _, _,  V_dic[n], V_left_dic[n]  =  bar_calculator(gamma_list_n, pars_dic=pars_dic, n=n)            
    return gamma_dic, err_dic, V_dic, V_left_dic


def looper(n_0, coeff_type, weight_type, weighting_scheme, is_round_1=True, pars_dic_dic=None):
    gamma_dic_exact = {}
    V_dic_exact = {}
    V_left_dic_exact = {}

    for n in range(2, n_max_exact + 1):
        print(f'\r\tCalculating n = {n}\t', end='')
        aa = range(1, n + 1)
        aa_w = np.array([f(i, weight_type=weight_type, n_0=n_0) for i in aa])

        perms = permutations(aa)
        gamma_list_n = []
        for perm in perms:
            bb = list(perm)
            w = w_calculator_permutations(bb=bb, n_0=n_0, aa_w = aa_w, weight_type=weight_type, weighting_scheme=weighting_scheme)
            gamma_list_n.append(gamma_permutations(bb=bb, w=w, coeff_type=coeff_type))

        if is_round_1:
            gamma_bar, _, _, _ =  bar_calculator(gamma_list_n)
            gamma_dic_exact[n] = gamma_bar
        else:
            pars_dic=pars_dic_dic['gamma'][coeff_type][weighting_scheme][weight_type][n_0]
            _, _, V_dic_exact[n], V_left_dic_exact[n]  =  bar_calculator(gamma_list_n, pars_dic=pars_dic, n=n)
    
    n_list = n_list_calc(coeff_type=coeff_type)
    gamma_dic_sample, err_dic_sample, V_dic_sample, V_left_dic_sample = sample_looper(n_0, coeff_type, weight_type, weighting_scheme, n_list, is_round_1=is_round_1, pars_dic_dic=pars_dic_dic)

    if is_round_1:
        return gamma_dic_exact, gamma_dic_sample, err_dic_sample, '', ''
    else:
        return '', '', '', V_dic_exact, V_left_dic_exact, V_dic_sample, V_left_dic_sample


def optimal_deg_calc(mse_dic_n_0):
    # ratios = [(i, mse_dic_n_0[i] / mse_dic_n_0.get(i - 1, 1), mse_dic_n_0[i] / mse_dic_n_0.get(i - 2, 1)) for i in mse_dic_n_0.keys()]
    # return max([d for d, r1, r2 in ratios if  r1 <= max_ratio1 and r2 <= max_ratio2])
    ratios = [(i, mse_dic_n_0[i] / mse_dic_n_0.get(i - 1, 10 ** 8), mse_dic_n_0.get(i + 1, 10 ** 8) / mse_dic_n_0.get(i - 1, 10 ** 8)) for i in mse_dic_n_0.keys()]
    d_opt = 1
    for d, r1, r2 in ratios:
        if  r1 <= max_ratio1 or r2 <= max_ratio2:
            d_opt = d
        else:
            break
    return d_opt

def regressor_final(dic_dic, opt_deg_dic, coeff_type, weighting_scheme, wtype, n_0, y_type='gamma'):
    deg = opt_deg_dic[y_type][coeff_type][weighting_scheme][wtype][n_0]
    if y_type == 'gamma':
        y_dic = dic_dic[coeff_type][weighting_scheme][wtype]['gamma_dic'][n_0]
        y_err_dic = dic_dic[coeff_type][weighting_scheme][wtype]['gamma_err_dic'][n_0]
    elif y_type == 'V':
        y_dic = dic_dic[coeff_type][weighting_scheme][wtype]['V_dic'][n_0]
        y_err_dic = None            
    elif y_type == 'V_left':
        y_dic = dic_dic[coeff_type][weighting_scheme][wtype]['V_left_dic'][n_0]
        y_err_dic = None 
    is_log = x_is_log(wtype, weighting_scheme, y_type)
    pars = regressor(y_dic=y_dic, deg=deg, is_log=is_log, y_err_dic=y_err_dic, is_train=False)
    return {'is_log': is_log, 'pars': pars}


def opt_dec_dic_fill(opt_deg_dic, dic_dic, max_deg, y_type='gamma'):    
    for coeff_type in dic_dic.keys():
        for weighting_scheme in dic_dic[coeff_type].keys():
            for wtype in dic_dic[coeff_type][weighting_scheme].keys():
                is_log = x_is_log(wtype, weighting_scheme, y_type)        
                if y_type == 'gamma':
                    y_dic = dic_dic[coeff_type][weighting_scheme][wtype]['gamma_dic']
                    y_err_dic = dic_dic[coeff_type][weighting_scheme][wtype]['gamma_err_dic']            
                elif y_type == 'V':
                    y_dic = dic_dic[coeff_type][weighting_scheme][wtype]['V_dic']
                    y_err_dic = None            
                elif y_type == 'V_left':
                    y_dic = dic_dic[coeff_type][weighting_scheme][wtype]['V_left_dic']
                    y_err_dic = None          
                
                n_0_list = list(y_dic.keys())
                for n_0 in n_0_list:
                    mse_dic_n_0 = {}
                    for deg in range(1, max_deg + 1):
                        if y_err_dic:
                            pars = regressor(y_dic = y_dic[n_0], deg=deg, is_log=is_log, y_err_dic = y_err_dic[n_0], is_train=True)
                        else:
                            pars = regressor(y_dic = y_dic[n_0], deg=deg, is_log=is_log, y_err_dic = None, is_train=True)                    
                        mse_dic_n_0[deg] = mse(y_dic[n_0], pars, is_log=is_log)
                    opt_deg = optimal_deg_calc(mse_dic_n_0)
                    opt_deg_dic[y_type][coeff_type][weighting_scheme][wtype][n_0] = opt_deg    
    return opt_deg_dic


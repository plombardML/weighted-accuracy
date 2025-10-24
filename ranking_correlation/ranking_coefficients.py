import json
import math
import numpy as np
from ranking_coeff_params import opt_deg_dic, epsilon_bound, epsilon, max_ranking_length, wtype_to_n_0_list
from params import *

# READ parameters
with open('ranking_correlation/estimation_results/gamma_dic_dic.json', 'r', encoding ='utf8') as json_file:
    gamma_dic_dic = json.load(json_file)
with open('ranking_correlation/estimation_results/V_dic_dic.json', 'r', encoding ='utf8') as json_file:
    V_dic_dic = json.load(json_file)
with open('ranking_correlation/estimation_results/pars_dic_dic.json', 'r', encoding ='utf8') as json_file:
    pars_dic_dic = json.load(json_file)


def f(x, weight_type, n_0=0):
    n_0 = int(n_0)
    if weight_type == '1':
        return 1 / x
    elif weight_type == '2':
        return 1 / (x + n_0) ** 2
    else:
        raise ValueError(f'Unexpected weight_type: {weight_type}')


def w_calculator(aa, bb, n_0, weight_type='1', weighting_scheme='add'):
    aa_w = np.array([f(x, weight_type=weight_type, n_0=n_0) for x in aa])
    bb_w = np.array([f(x, weight_type=weight_type, n_0=n_0) for x in bb])
    if weighting_scheme == 'add':
        return aa_w + bb_w
    elif weighting_scheme == 'mult':
        return aa_w * bb_w
    else:
        raise ValueError(f"Unexpected weighting_scheme: {weighting_scheme}")


def spearman(aa, bb, w):
    n = len(bb)
    wsum = sum(w)
    aam = sum([w[i] * aa[i] for i in range(n)]) / wsum
    bbm = sum([w[i] * bb[i] for i in range(n)]) / wsum
    ssa = math.sqrt(sum([w[i] * (aa[i] ** 2 - aam ** 2) for i in range(n)]) / wsum)
    ssb = math.sqrt(sum([w[i] * (bb[i] ** 2 - bbm ** 2) for i in range(n)]) / wsum)
    coeff = sum([w[i] * (aa[i] - aam) * (bb[i] - bbm) for i in range(n)]) / (wsum * ssa * ssb)
    return coeff


def kendall(aa, bb, w):
    n = len(bb)
    num = 0
    den = 0
    for i in range(0, n):
        for j in list(range(0, i)) + list(range(i + 1, n)):
            x = w[i] * w[j]
            num += x * np.sign((aa[j] - aa[i]) * (bb[j] - bb[i]))
            den += x        
    return num / den


def gamma_calc(aa, bb, w, coeff_type):
    if coeff_type == 'spearman':
        return spearman(aa, bb=bb, w=w)
    elif coeff_type == 'kendall':
        return kendall(aa, bb=bb, w=w)
    else:
        raise ValueError(f'Unknown coeff_type: {coeff_type}') 


def n_to_x(n):
    return 1 / int(n)


def n_to_x_log(n):
    return 1 / math.log(int(n))


def poly_estimator(n, pars, is_log):
    if is_log:
        x = n_to_x_log(n)
    else:
        x = n_to_x(n)
    return sum([x ** i * pars[i] for i in range(len(pars))])


def x_is_log(wtype, weighting_scheme, y_type='gamma'):
    if y_type == 'gamma':
        if wtype == '1' or weighting_scheme == 'mult':
            return True
        else:
            return False
    else:
        if wtype == '2' and weighting_scheme == 'mult':
            return True
        else:
            return False






def monotonicity_requirement(gamma_bar, V, V_left):
    A = 2 * V_left - V * (1 + gamma_bar)
    B = ((1 - gamma_bar ** 2) * V - 4 * gamma_bar * V_left - (1 - gamma_bar ** 2) ** 2) / A
    C = (2 * (1 + gamma_bar ** 2) * V_left - (1 - gamma_bar ** 2) * V) / A
    D = (C - 2 * (1 - gamma_bar)) / (2 * (1- gamma_bar) - B)
    E = - C / B
    F = (2 * (1 + gamma_bar) - C) / (2 * (1 + gamma_bar) + B)
    
    # bound 1
    bound_1_low = -1
    bound_1_up = 1
    if 2 * (1 - gamma_bar) - B > 0:
        bound_1_low = D
    elif 2 * (1 - gamma_bar) - B < 0:
        bound_1_up = D
    else:
        if abs(C - 2 * (1 - gamma_bar)) > epsilon_bound:
            ValueError('Bound Consistency Error for bound 1')

    # bound 2
    bound_2_low = -1
    bound_2_up = 1
    if B > 0:
        bound_2_low = E
    elif B < 0:
        bound_2_up = E
    else:
        if abs(C) > epsilon_bound:
            ValueError('Bound Consistency Error for bound 2')

    # bound 3
    bound_3_low = -1
    bound_3_up = 1
    if 2 * (1 + gamma_bar) + B > 0:
        bound_3_up = F
    elif 2 * (1 + gamma_bar) + B < 0:
        bound_3_low = F
    else:
        if abs(C - 2 * (1 + gamma_bar)) > epsilon_bound:
            ValueError('Bound Consistency Error for bound 3')

    # bounds combination
    bound_low = max(bound_1_low, bound_2_low, bound_3_low)
    bound_up = min(bound_1_up, bound_2_up, bound_3_up)
    if bound_up < bound_low:
        ValueError(f'Bound Consistency Error. bound_low: {bound_low}, bound_up: {bound_up}')
    g_0 = 0
    if bound_low > g_0:
        g_0 = bound_low
    elif bound_up < g_0:
        g_0 = bound_up

    g_1 = (g_0 * B + C) / (1 - gamma_bar ** 2)
    return g_0, g_1


def p_params_calc(coeff_type, weighting_scheme, wtype, n_0, n):
    if n <= n_max_exact:
        gamma_bar = gamma_dic_dic[coeff_type][weighting_scheme][wtype]['gamma_dic_exact'][n_0][str(n)]
        V = V_dic_dic[coeff_type][weighting_scheme][wtype]['V_dic_exact'][n_0][str(n)]
        V_left = V_dic_dic[coeff_type][weighting_scheme][wtype]['V_left_dic_exact'][n_0][str(n)]
    else:
        # calculate gamma_bar
        is_log = pars_dic_dic['gamma'][coeff_type][weighting_scheme][wtype][n_0]['is_log']
        pars = pars_dic_dic['gamma'][coeff_type][weighting_scheme][wtype][n_0]['pars']
        gamma_bar = poly_estimator(n, pars, is_log)
        
        # calculate V
        is_log = pars_dic_dic['V'][coeff_type][weighting_scheme][wtype][n_0]['is_log']
        pars = pars_dic_dic['V'][coeff_type][weighting_scheme][wtype][n_0]['pars']
        V = poly_estimator(n, pars, is_log)
        
        # calculate V_left
        is_log = pars_dic_dic['V_left'][coeff_type][weighting_scheme][wtype][n_0]['is_log']
        pars = pars_dic_dic['V_left'][coeff_type][weighting_scheme][wtype][n_0]['pars']
        V_left = poly_estimator(n, pars, is_log)
    return gamma_bar, V, V_left


def g_params_calc(gamma_bar, V, V_left):
    diff = V_left - V * (1 + gamma_bar) / 2

    if abs(diff) < epsilon:
        g_0 = (V * gamma_bar) / (1 - V - gamma_bar ** 2)
        m = min(1 - gamma_bar - V, 1 + gamma_bar - V)
        if m < 0:
            raise ValueError(f'Found negative value  : 1 - gamma_bar-V: {1 - gamma_bar - V}, 1 + gamma_bar - V: {1 + gamma_bar - V}')

        bound = 2 * m / (1 - gamma_bar ** 2 - V)
        g_1 = 1
        if bound < 1:
            g_1 = bound
    else:
        g_0, g_1 = monotonicity_requirement(gamma_bar=gamma_bar, V=V, V_left=V_left)
    
    g_2 = - (1 + g_0) / (1 + gamma_bar) ** 2 + g_1 / (1 + gamma_bar)
    h_2 = (1 - g_0) / (1 - gamma_bar) ** 2 - g_1 / (1 - gamma_bar)

    return g_0, g_1, g_2, h_2


def g_calc(gamma, g_0, g_1, g_2, h_2, gamma_bar):
    if gamma < gamma_bar:
        g = g_0 + g_1 * (gamma - gamma_bar) + g_2 * (gamma - gamma_bar) ** 2
    else:
        g = g_0 + g_1 * (gamma - gamma_bar) + h_2 * (gamma - gamma_bar) ** 2
    return g


def standard_gamma_calc(coeff_type: str, weighting_scheme: str, wtype: int, aa: list, bb: list, n_0: int):
    """
    Compute a standardized gamma value based on ranking correlation and weighting parameters.

    This function calculates the standardized gamma coefficient for two ranking vectors,
    using a specified weighting scheme and correlation type (Spearman or Kendall).
    It validates the configuration and applies weighting and gamma computation accordingly.

    Parameters
    ----------
    coeff_type : {'spearman', 'kendall'}
        Type of rank correlation coefficient to use.
    weighting_scheme : {'add', 'mult'}
        The method used for computing weights: additive or multiplicative.
    wtype : {1, 2}
        An integer identifier for the weight type. wtype=1 corresponds to weighting function $f(i)=1/i$, wtype=2 corresponds to $f(i)=1/(i+n_0)^2$.
    aa : list of float
        First ranking vector. Must be the same length as `bb`.
    bb : list of float
        Second ranking vector. Must be the same length as `aa`.
    n_0 : int
        Parameter that further specifies the weigthing function. Must match allowed values for given `wtype`.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        
        - 'standard_gamma' : float
            The standardized gamma value (final output).
        - 'gamma' : float
            The raw gamma coefficient.
        - 'gamma_avg' : float
            The expected value of gamma.
        - 'V' : float
            The variance of gamma.
        - 'V_left' : float
            Left-side component of the variance of gamma.

    Raises
    ------
    ValueError
        If any parameter is outside the acceptable set of values, or if input lists are mismatched in length.

    Examples
    --------
    >>> standard_gamma_calc(
    ...     coeff_type='spearman',
    ...     weighting_scheme='add',
    ...     wtype=1,
    ...     aa=[1, 2, 3],
    ...     bb=[3, 2, 1],
    ...     n_0=2
    ... )
    {'standard_gamma': ..., 'gamma': ..., 'gamma_avg': ..., 'V': ..., 'V_left': ...}
    """

    if coeff_type not in ('spearman', 'kendall'):
        raise ValueError(f'Unacceptable coeff_type={coeff_type}. Acceptable values are \'spearman\', \'kendall\'')    
    if weighting_scheme not in ('add', 'mult'):
        raise ValueError(f'Unacceptable weighting_scheme={weighting_scheme}. Acceptable values are \'add\', \'mult\'')
    if wtype in (1, 2):
        wtype = str(wtype)
        n_0_list = wtype_to_n_0_list[wtype]
    else:
        raise ValueError(f'Unacceptable wtype={wtype}. Acceptable values are 1, 2')
    n = len(aa)
    if len(bb) != n:
        raise ValueError('aa and bb should have the same length')
    if n_0 in n_0_list:
        n_0 = str(n_0)
    else:
        raise ValueError(f'Unacceptable n_0={n_0}. Acceptable values for n_0 when wtype={wtype} are {n_0_list}')

    is_log_gamma = x_is_log(wtype=wtype, weighting_scheme=weighting_scheme, y_type='gamma')
    is_log_V = x_is_log(wtype=wtype, weighting_scheme=weighting_scheme, y_type='V')
    if is_log_gamma or is_log_V:
        n_max = max_ranking_length[coeff_type]
        if n > n_max:
            raise ValueError(f'Ranking length outside of known range for these parameters. Please choose rankings of length < {n_max} or change parameters')

    w = w_calculator(aa=aa, bb=bb, n_0=n_0, weight_type=wtype, weighting_scheme=weighting_scheme)
    gamma = gamma_calc(aa=aa, bb=bb, w=w, coeff_type=coeff_type)

    gamma_bar, V, V_left = p_params_calc(coeff_type=coeff_type, weighting_scheme=weighting_scheme, wtype=wtype, n_0=n_0, n=n)
    g_0, g_1, g_2, h_2 = g_params_calc(gamma_bar=gamma_bar, V=V, V_left=V_left)

    g = float(g_calc(gamma, g_0, g_1, g_2, h_2, gamma_bar))

    out = {
        'standard_gamma': g,
        'gamma': float(gamma),
        'gamma_avg': gamma_bar,
        'V': V,
        'V_left': V_left
    }
    
    return out
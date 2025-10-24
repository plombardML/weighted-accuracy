import json
import math
import os
import numpy as np
from scipy import stats

from WA_params import *

from ranking_coefficients import standard_gamma_calc

def beta_par_calculator(N_tot, C_frac):
    """Calculate Beta distribution parameters for informed H and EWA metrics.
    
    Args:
        N_tot (int): Total number of samples
        C_frac (float): Cost fraction parameter
        
    Returns:
        tuple: Beta distribution parameters (a, b)
    """
    a_min = 2
    c_avg = 1 - C_frac
    M = P_eff * R_avg * (1 - C_frac)
    sigma_c = M * sigma_R / (math.sqrt(N_tot) * P_eff * R_avg ** 2)
    a = (1 - c_avg) * c_avg ** 2 / sigma_c - c_avg
    a = max(a, a_min)
    b = a / c_avg - a
    return a, b


class Metrics:
    """Container for storing and managing various classification metrics."""
    
    def __init__(self):
        """Initialize metrics dictionary with empty lists for each metric."""
        self.metrics_dic = {}
        for metric in metrics_of_interests:
            self.metrics_dic[metric] = []
        self.metrics_dic['TCC'] = []    
    
    def append_to_list(self, metric_name, entry):
        """Append a value to the specified metric's list.
        
        Args:
            metric_name (str): Name of the metric
            entry: Value to append
            
        Raises:
            ValueError: If metric_name is not recognized
        """
        if metric_name not in  self.metrics_dic.keys():
            raise ValueError(f'Unknown metric_name: {metric_name}. Known values are {self.metrics_dic.keys()}.')
        else:
            self.metrics_dic[metric_name].append(entry)
    
    def get_list(self, metric_name):
        """Get the list of values for a specified metric.
        
        Args:
            metric_name (str): Name of the metric
            
        Returns:
            list: Values for the specified metric
            
        Raises:
            ValueError: If metric_name is not recognized
        """
        if metric_name not in  self.metrics_dic.keys():
            raise ValueError(f'Unknown metric_name: {metric_name}. Known values are {self.metrics_dic.keys()}.')
        else:
            return self.metrics_dic[metric_name]


class Dataset:
    """Dataset class for binary classification with cost-sensitive evaluation."""
    
    def __init__(self, P, N):
        """Initialize dataset with positive and negative sample counts.
        
        Args:
            P (int): Number of positive samples
            N (int): Number of negative samples
        """
        self.P = P if P > 0 else epsilon
        self.N = N if N > 0 else epsilon
        self.N_tot = P + N
        self.ratio_plus = P / self.N_tot
        self.C_FP = None
        self.C_FN = None
        self.w = None
        self.FP = None
        
    def set_cost(self, C_FP, C_FN):
        """Set the cost parameters for false positives and false negatives.
        
        Args:
            C_FP (float): Cost of false positive
            C_FN (float): Cost of false negative
        """
        self.C_FP = C_FP
        self.C_FN = C_FN
    
    def set_confusion_matrix(self, FN, FP):
        """Set confusion matrix values and calculate all derived metrics.
        
        Args:
            FN (int): Number of false negatives
            FP (int): Number of false positives
        """
        self.FN = FN
        self.FP = FP
        self.TN = self.N - FP
        self.TP = self.P - FN
        self.prec = self.TP / (self.TP + self.FP + epsilon)
        self.rec = self.TP / self.P
        self.spec = self.TN / self.N
        self.NPV = self.TN / (self.TN + self.FN + epsilon)
        self.accuracy = (self.TP + self.TN) / (self.N_tot)
        self.F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)       
        self.kappa = (2 * (self.TP * self.TN - self.FN * self.FP) / ((self.TP + self.FP) * self.N + self.P * (self.FN + self.TN)) + epsilon)
 
        # P4 metric
        self.P4 = (4*self.TP*self.TN) / (4*self.TP*self.TN + (self.TP+self.TN)*(self.FP+self.FN) + epsilon)
       
        # ROC-AUC for a single parametrization model -> area aunder segments
        self.ROC_AUC = (self.TP / self.P + self.TN / self.N) / 2

        # B-ROC for a single parametrization model -> area aunder segments
        self.B_ROC = (self.TP / self.P + self.TP / (self.FP + self.TP + epsilon)) / 2
        
        # Class balance accuracy 
        self.CBA = 1/2 * (self.TP / max(self.P, self.TP + self.FP) + self.TN / max(self.N, self.TN + self.FN))
      
        # IAM 
        self.IAM = 1/2 * ((self.TP - max(self.FP, self.FN)) / max(self.P, self.TP + self.FP) + (self.TN - max(self.FP, self.FN)) / max(self.N, self.TN + self.FN))
        
        # Matthews Correlation Coefficient
        self.MCC = (self.TP * self.TN - self.FP * self.FN) / (math.sqrt((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN)) + epsilon) 
                
        # informedness and maskedness 
        self.informedness = self.TP / self.P - self.FP / self.N
        self.markedness = self.TP / (self.TP +  self.FP + epsilon) - self.FN / (self.TN + self.FN + epsilon)

        # G-mean 
        self.G_mean = math.sqrt((self.TP * self.TN)/ (self.P * self.N))
        
        # # ACD
        # TCC_avg =  self.FP * self.C_FP + self.FN * self.C_FN
        # TCC_avg_max = self.N * self.C_FP + self.P * self.C_FN
        # self.ACD = math.sqrt(( 1 - self.accuracy) ** 2 + (TCC_avg / S(TCC_avg_max)) ** 2)
        

        #### Cost based metrics        
        self.cost = self.C_FN * self.FN + self.C_FP * self.FP
        self.max_cost = self.C_FN * self.P + self.C_FP * self.N    
        self.w_cost = self.C_FN /(self.C_FP + self.C_FN)

        # Accuracy-Cost Distance
        self.ACD = math.sqrt((1 - self.accuracy) ** 2 + (self.cost / self.max_cost) ** 2)

        # Weighted Relative Accuracy 
        self.WRA = (4 * self.N / self.P * self.C_FP / self.C_FN * (self.TP / self.P - self.FP / self.N)) / (1 + self.N / self.P * self.C_FP / self.C_FN)**2
        
    def set_w(self, w=None):
        """Set weight parameter and calculate weighted metrics.
        
        Args:
            w (float, optional): Weight parameter. If None, uses cost-based weight.
        """
        if w:
            self.w = w
        else:
            self.w = self.w_cost
        
        self.WCA = self.w * self.TP / self.P + (1 - self.w) * self.TN / self.N
       
        self.WA = (self.w*self.TP + (1-self.w)*self.TN) / (self.w*(self.TP+self.FN) + (1-self.w)*(self.TN+self.FP))

    
    def get_WA(self):
        """Get Weighted Accuracy."""
        return self.WA
        
    def get_rescaled_WA(self):
        """Get rescaled Weighted Accuracy."""
        return self.rescaled_WA
        
    def get_P4(self):
        """Get P4 metric."""
        return self.P4
        
    def get_ROC_AUC(self):
        """Get ROC-AUC score."""
        return self.ROC_AUC
        
    def get_B_ROC(self):
        """Get B-ROC score."""
        return self.B_ROC
        
    def get_recall(self):
        """Get recall (sensitivity)."""
        return self.rec
        
    def get_precision(self):
        """Get precision."""
        return self.prec
        
    def get_F1(self):
        """Get F1 score."""
        return self.F1
        
    def get_WCA(self):
        """Get Weighted Class Accuracy."""
        return self.WCA
        
    def get_WRA(self):
        """Get Weighted Relative Accuracy."""
        return self.WRA
        
    def get_cm(self):
        """Get confusion matrix values.
        
        Returns:
            tuple: (TP, FP, TN, FN)
        """
        return self.TP, self.FP, self.TN, self.FN

    def get_H_complete(self, a=2, b=2, num_points=1000):
        """Calculate H-measure using Beta distribution integration.
        
        Args:
            a (float): Beta distribution alpha parameter
            b (float): Beta distribution beta parameter
            num_points (int): Number of integration points
            
        Returns:
            float: H-measure value
        """
        c_vals = np.linspace(0, 1, num_points)[1:-1]
        u_c = stats.beta.pdf(c_vals, a, b)  # Beta(a, b) distribution

        # Evaluate TCC and TCC_max across c_vals
        TCC_vals_S = self.FP * c_vals + self.FN * (1 - c_vals)
        TCC_max_vals_S = self.N * c_vals + self.P * (1 - c_vals)

        # Integrate using trapezoidal rule
        num = np.trapezoid(TCC_vals_S * u_c, c_vals)
        den = np.trapezoid(TCC_max_vals_S * u_c, c_vals)

        # Handle division by zero
        if den == 0:
            return 0.0
        return 1 - (num / den)

    
    def get_EWA(self, a=2, b=2, num_points=1000):
        """Calculate Expected Weighted Accuracy using Beta distribution integration.
        
        Args:
            a (float): Beta distribution alpha parameter
            b (float): Beta distribution beta parameter
            num_points (int): Number of integration points
            
        Returns:
            float: EWA value
        """
        c_vals = np.linspace(0, 1, num_points)[1:-1]
        u_c = stats.beta.pdf(c_vals, a, b)  # Beta(a, b) distribution

        # Evaluate WA across c_vals (w = 1 - c)
        num_list = self.TP * (1 - c_vals) + self.TN * c_vals
        den_list = self.P * (1 - c_vals) +  self.N * c_vals
        
        # Integrate using trapezoidal rule
        out = np.trapezoid(u_c * num_list / den_list, c_vals)
        return out


def compare_with_TCC(input_list, cost_list, metric='', inverse=True, weight=False):
    """Compute correlation between metrics and Total Cost of Classification (TCC).
    
    Args:
        input_list (list): Metric values to compare
        cost_list (list): TCC values for comparison
        metric (str): Name of the metric being compared
        inverse (bool): Whether to use inverse cost ranking
        weight (bool): Whether to use weighted correlation
        
    Returns:
        pd.DataFrame: DataFrame with metric name and correlation coefficient
    """

    ranks = stats.rankdata(input_list)
    cost_ranks = stats.rankdata(cost_list)
    inverse_cost_ranks = stats.rankdata([-c for c in cost_list])
    if inverse:
        baseline_ranks = inverse_cost_ranks
    else:
        baseline_ranks = cost_ranks
    if weight:
        s = standard_gamma_calc(coeff_type='spearman', weighting_scheme='add', wtype=2, aa=ranks, bb=baseline_ranks, n_0=2)['standard_gamma']
    else:
        s = stats.spearmanr(ranks, baseline_ranks)[0]        
    
    return pd.DataFrame({'metric': [metric], 's': [s]})




def generate_metrics_and_compare(D, N, P, M, R_list, n_samples=10, weight=False):
    """Generate metrics across different prediction scenarios and compare with TCC.
    
    Args:
        D (Dataset): Dataset object
        N (int): Number of negative samples
        P (int): Number of positive samples
        M (float): Cost multiplier
        R_list (list): List of cost values
        n_samples (int): Number of sampling iterations
        weight (bool): Whether to use weighted correlation
        
    Returns:
        tuple: (average_correlations_dict, std_correlations_dict)
    """
    df_list_by_sample = []
    for s in range(n_samples):
        Metric = Metrics()    
        indeces_pos = rng_utils.choice(range(D.N_tot), P, replace=False)
        for N_pred_pos in range(0, D.N_tot + 1):
            # indices of predicted positives
            indeces_pred_pos = rng_utils.choice(range(D.N_tot), N_pred_pos, replace=False)
            cost_FN_list = [R_list[i] for i in indeces_pos if i not in indeces_pred_pos]
            FN = len(cost_FN_list)
            FP = len([i for i in indeces_pred_pos if i not in indeces_pos])
            TN = N - FP
            TP = P - FN
        
            TCC = M * (FP - FN) + P_eff * sum(cost_FN_list)

            D.set_confusion_matrix(FN=FN, FP=FP)
            Metric.append_to_list('precision', D.prec)
            Metric.append_to_list('recall', D.rec)
            Metric.append_to_list('specificity', D.spec)
            Metric.append_to_list('NPV', D.NPV)
            Metric.append_to_list('accuracy', D.accuracy)
            Metric.append_to_list('F1', D.F1)
            Metric.append_to_list('kappa', D.kappa)
            Metric.append_to_list('P4', D.P4)
            Metric.append_to_list('ROC-AUC', D.ROC_AUC)
            Metric.append_to_list('CBA', D.CBA)
            Metric.append_to_list('IAM', D.IAM)
            Metric.append_to_list('MCC', D.MCC)
            Metric.append_to_list('informedness', D.informedness)
            Metric.append_to_list('markedness', D.markedness)
            Metric.append_to_list('ACD', D.ACD) 
            Metric.append_to_list('B-ROC', D.B_ROC)
            Metric.append_to_list('WRA', D.WRA)
            D.set_w()
            Metric.append_to_list('WCA', D.WCA)
            Metric.append_to_list('WA', D.WA)
            Metric.append_to_list('H', D.get_H_complete())
            
            a, b = beta_par_calculator(N_tot=D.N_tot, C_frac=D.w_cost)
            Metric.append_to_list('H informed', D.get_H_complete(a=a, b=b))
            Metric.append_to_list('EWA', D.get_EWA(a=a, b=b))
            
            Metric.append_to_list('G-mean', D.G_mean)
            Metric.append_to_list('TCC', TCC)
 
    
        df_list = []
        for metric in Metric.metrics_dic.keys():
            inverse = True
            if metric in ('ACD', 'C-score'):
                inverse = False
            df0 = compare_with_TCC(input_list=Metric.metrics_dic[metric], cost_list=Metric.get_list('TCC'), metric=metric, inverse=inverse, weight=weight)
            df_list.append(df0)
        df = pd.concat(df_list)
        df_list_by_sample.append(df)
        
    dic_avg = {}
    dic_std = {}
    for metric in Metric.metrics_dic.keys():
        corr_list = [df_m[df_m.metric == metric]['s'][0] for df_m in df_list_by_sample]
        dic_avg[metric] = np.mean(corr_list)
        dic_std[metric] = np.std(corr_list)
    
    return dic_avg, dic_std


def calculator(weight):
    """Calculate correlation heatmap data for different dataset configurations.
    
    Args:
        weight (bool): Whether to use weighted correlation calculations
        
    Returns:
        tuple: (average_metrics_dataframes, std_metrics_dataframes)
    """
    Metric = Metrics()
    metric_data_avg = {m: [] for m in Metric.metrics_dic.keys()}  
    metric_data_std = {m: [] for m in Metric.metrics_dic.keys()}  
    
    print(f'N_tot: {N_tot}, n_samples: {n_samples}')
    print(f'frac_plus_list: {frac_plus_list}')
    print(f'C_frac_list: {C_frac_list}')
    # R_list = rng_utils.choice(RCV_list, N_tot, replace=False)

    for frac_plus in frac_plus_list:
        P = int(round(N_tot * frac_plus))
        N = N_tot - P
        for C_frac in C_frac_list:  
            print(f'frac_plus={frac_plus}, C_frac={C_frac}        ', end='\r')
            D = Dataset(P=P, N=N)
            D.set_cost(C_FN=C_frac / (1 - C_frac), C_FP=1)
            M = P_eff * R_avg * (1 - C_frac)
            dic_avg, dic_std = generate_metrics_and_compare(D, N, P, M, R_list=R_list, n_samples=n_samples, weight=weight)
            for m in Metric.metrics_dic.keys():
                if m in dic_avg.keys():  
                    metric_data_avg[m].append((P / N_tot, C_frac, magnifying_factor * dic_avg[m]))  
                    metric_data_std[m].append((P / N_tot, C_frac, magnifying_factor * dic_std[m]))      
    
    metric_dfs_avg = {m: pd.DataFrame(data, columns=['P', 'cost', 'value']) for m, data in metric_data_avg.items()}
    metric_dfs_std = {m: pd.DataFrame(data, columns=['P', 'cost', 'value']) for m, data in metric_data_std.items()}

    filename = f'{out_path_0}/weight_{weight}/avg/'
    os.makedirs(filename, exist_ok=True)
    for k in metric_data_avg.keys():
        metric_dfs_avg[k].to_csv(filename + f'{k}.csv')
    
    filename = f'{out_path_0}/weight_{weight}/std/'
    os.makedirs(filename, exist_ok=True)
    for k in metric_data_std.keys():
        metric_dfs_std[k].to_csv(filename + f'{k}.csv')
    
    os.makedirs(out_path_0, exist_ok=True)
    
    with open(f'{out_path_0}/params.json', 'w') as f:
        json.dump(params, f)
    
    return metric_dfs_avg, metric_dfs_std


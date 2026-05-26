# WA PARAMS

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rng_utils = np.random.default_rng(142)


####
# sampling params
n_samples = 100
N_tot = 200
C_frac_list = [0.01] + [i / 10 for i in range(1, 10)] + [0.99]
frac_plus_list = [0.01] + [i / 10 for i in range(1, 10)] + [0.99]

# C_frac_list = [0.01] + [i / 10 for i in range(1, 10)] + [0.99]
# frac_plus_list = [0.01] + [i / 10 for i in range(1, 10)] + [0.99]



#########
# CHURN PARAMS

####
# Randomly extract N_tot revenues from Kaggle dataset List of revenues

CUSTOMER_CHURN_DATASET_PATH = './datasets/Customer_Churn_Dataset.csv'
df_customers = pd.read_csv(CUSTOMER_CHURN_DATASET_PATH, sep=';', )
RCV_list = [float(x.replace(',', '.')) for x in df_customers['MonthlyCharges']]
R_list = rng.choice(RCV_list, N_tot, replace=False)


# cost params
P_eff = 0.25 # probability that the retention measure is effective
R_avg = np.mean(R_list)
sigma_R = np.std(R_list)
R_rescaled_list = [P_eff * x for x in sorted(R_list)]


#########
# CHURN WITH EXTREME STATISTICS

massive_customers_fraction_lst = [0.01, 0.02, 0.05, 0.1, 0.2]
revenue_fraction_lst = [0.2, 0.4, 0.6, 0.8, 0.99]
n_mcfl = len(massive_customers_fraction_lst)
n_rfl = len(revenue_fraction_lst)

R_list_extreme_array = [[None for i in range(n_mcfl)] for j in range(n_rfl)]
R_avg_extreme_array = [[None for i in range(n_mcfl)] for j in range(n_rfl)]
sigma_R_extreme_array = [[None for i in range(n_mcfl)] for j in range(n_rfl)]
R_rescaled_list_extreme_array = [[None for i in range(n_mcfl)] for j in range(n_rfl)]

R_list_sorted = sorted(R_list, reverse=True)
for i in range(n_mcfl):
    for j in range(n_rfl):
        massive_customers_fraction = massive_customers_fraction_lst[i]
        revenue_fraction = revenue_fraction_lst[j]
        
        massive_customers_number = int(round(N_tot * massive_customers_fraction))
        revenue_fraction_others = 1 - revenue_fraction
        R_list_massive = R_list_sorted[:massive_customers_number]
        R_list_others = R_list_sorted[massive_customers_number:]
        
        R_total = sum(R_list_others) / revenue_fraction_others
        R_massive_customers = R_total * revenue_fraction
        R_massive_customers_initial = sum(R_list_massive)
        R_factor_massive_customers = R_massive_customers / R_massive_customers_initial
        R_list_massive = [r * R_factor_massive_customers for r in R_list_massive]        
        
        R_list_extreme = np.array(R_list_massive + R_list_others)
        
        # cost params
        R_avg_extreme = np.mean(R_list_extreme)
        sigma_R_extreme = np.std(R_list_extreme)
        R_rescaled_list_extreme = [P_eff * x for x in sorted(R_list_extreme)]

        R_list_extreme_array[i][j] = R_list_extreme
        R_avg_extreme_array[i][j] = R_avg_extreme
        sigma_R_extreme_array[i][j] = sigma_R_extreme
        R_rescaled_list_extreme_array[i][j] = R_rescaled_list_extreme


#########
# CREDIT SCORING PARAMS
# Parameters taken from paper "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring"
cs_params = {
    'l': 24, 'intcf': 0.0294/12, 'intr': 0.0479/12, 'k': 3, 'Clmax': 25000, 'Lgd': 0.75,
}

CREDIT_SCORING_DATASET_PATH = f'./datasets/credit_scoring.csv'
df_credit = pd.read_csv(CREDIT_SCORING_DATASET_PATH)
# Column Mapping
cs_col_map = {
    'income': 'MonthlyIncome',
    'debt': 'DebtRatio', 
    'target': 'SeriousDlqin2yrs'
}
# SeriousDlqin2yrs: Person experienced 90 days past due delinquency or worse (Target variable / label)
# Delinquency: Being late on loan/credit payments


#########

out_path_0 = f'./results/'

epsilon = 10 ** (-9) # regularization parameter to avoid division by zero

####
# plot params
magnifying_factor = 10 # to avoid '0.' characters in annotations
plot_n_rows = 5
plot_n_cols = 5


params = {
    'P_eff': P_eff,
    'R_avg': R_avg,
    'sigma_R': sigma_R,
    'n_samples': n_samples,
    'N_tot': N_tot,
    'epsilon': epsilon
}

metrics_of_interests = [
    'accuracy',
    'CBA',
    'IAM',
    'H',
    'WCA',
    'kappa',
    'informedness',
    'ROC-AUC',
    'WRA',
    'MCC',
    'markedness',
    'precision',
    'NPV',
    'P4',
    'G-mean',
    'jaccard',
    'F1',
    'recall',
    'B-ROC',
    'specificity',
    'ACD',
    'WA',   
    'H informed',
    'EWA'
]

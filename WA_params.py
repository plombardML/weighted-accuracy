# WA PARAMS

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rng_utils = np.random.default_rng(142)


####
# sampling params
n_samples = 100
N_tot =200
C_frac_list = [0.01] + [i / 10 for i in range(1, 10)] + [0.99]
frac_plus_list = [0.01] + [i / 10 for i in range(1, 10)] + [0.99]



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
    'F1',
    'recall',
    'B-ROC',
    'specificity',
    'ACD',
    'WA',   
    'H informed',
    'EWA'    
]

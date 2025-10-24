# PARAMS

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

####
# sampling params
n_samples = 100
N_tot =200
C_frac_list = [0.01] + [i / 10 for i in range(1, 10)] + [0.99]
frac_plus_list = [0.01] + [i / 10 for i in range(1, 10)] + [0.99]


####
# Randomly extract N_tot revenues from Kaggle dataset List of revenues
df_customers = pd.read_csv('./customer_values/Customer_Churn_Dataset.csv', sep=';', )
RCV_list = [float(x.replace(',', '.')) for x in df_customers['MonthlyCharges']]
R_list = rng.choice(RCV_list, N_tot, replace=False)


rng_utils = np.random.default_rng(142)


# cost params
P_eff = 0.25 # probability that the retention measure is effective
R_avg = np.mean(RCV_list)
sigma_R = np.std(RCV_list)


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
    'recall',
    'precision',
    'specificity',
    'NPV',
    'F1',
    'informedness',
    'markedness',
    'MCC',
    'kappa',
    'G-mean',
    'ROC-AUC',
    'CBA',
    'IAM',
    'P4',
    'B-ROC',
    'WCA',
    'WRA',
    'ACD',
    'H',
    'H informed',
    'WA',   
    'EWA'    
]


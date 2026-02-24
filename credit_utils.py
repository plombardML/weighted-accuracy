import pandas as pd
import numpy as np

### Basic functions to compute costs. 
# Derived from paper "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring"

# Monthly payment calculation
def A(Cl, intr, l):
    return Cl * (intr * (1 + intr)**l) / ((1 + intr)**l - 1)

# Present value calculation
def PV(A_val, intcf, l):
    return A_val / intcf * (1 - 1/((1 + intcf)**l))

# Payment ratio calculation
def Pm(debt_ratio, monthly_income, intr, l, k):
    A_max = A(k * monthly_income, intr, l)
    debt_ratio_clipped = np.clip(debt_ratio, 0, 1)
    return np.minimum(A_max / monthly_income, 1 - debt_ratio_clipped)

# Credit limit with debt constraint
def Clmax_debt(debt_ratio, monthly_income, intr, l, k):
    pm = Pm(debt_ratio, monthly_income, intr, l, k)
    A_val = monthly_income * pm
    return PV(A_val, intr, l)

# Credit limit calculation
def Cl(monthly_income, debt_ratio, k, Clmax, intr, l):
    cl_income = k * monthly_income
    cl_debt = Clmax_debt(debt_ratio, monthly_income, intr, l, k)
    Clmax = [Clmax]*len(cl_debt)
    return np.minimum.reduce([cl_income, Clmax, cl_debt])

### Utility functions to obtain costs

def compute_credit_scoring_costs(df, income_col, debt_col, target_col, params):
    p = params

    # Data Cleaning
    for col in [income_col, target_col]:
        df[col] = pd.to_numeric(df[col].replace('N', np.nan), errors='coerce')

    # Calculate credit limits
    df['Cl'] = Cl(df[income_col], df[debt_col], p['k'], p['Clmax'], p['intr'], p['l'])

    # Profit per customer
    A_vals = A(df['Cl'], p['intr'], p['l'])
    PV_vals = PV(A_vals, p['intcf'], p['l'])
    df['r'] = PV_vals - df['Cl']
    return df


def df_UCC_calculator_credit(df, C_frac, P, N):
    N_tot = P + N
    df_UCC_credit = df.dropna().sample(N_tot, replace=False, ignore_index=True, random_state=242)

    # Prior probabilities
    r_plus = P / N_tot
    pi1 = r_plus  # positive rate (default)
    pi0 = 1 - pi1  # negative rate (no default)

    # Alternative customer cost
    avg_r = df_UCC_credit['r'].mean()
    avg_Cl = df_UCC_credit['Cl'].mean()
        
    # Loss given default parameter tuned to accomodate C_frac; 
    # note: Lgd is positive only if C_frac < 1 / (1 + r_plus)
    if C_frac < 1 / (1 + r_plus):
        Lgd = (r_plus * avg_r * C_frac) / (avg_Cl * (1 - (r_plus + 1) * C_frac))
        
        C_fp_a = -avg_r * pi0 + avg_Cl * Lgd * pi1

        # Cost of False Negative
        df_UCC_credit['C_FN'] = df_UCC_credit['Cl'] * Lgd

        # Cost of False Positive
        # enforce positive costs
        df_UCC_credit['C_FP'] = [max(0, x) for x in df_UCC_credit['r'] + C_fp_a]


        return df_UCC_credit[['C_FN', 'C_FP']]
    else:
        return None


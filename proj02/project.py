# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans_df):
    
    def drop_months(term_len):
        return term_len.split(" ")[1]
    
    def fix_emp(employment):
        employment = employment.lower()
        employment = employment.strip()
        if employment == "rn":
            employment = "registered nurse"
        return employment
        
    
    loans_df['issue_d'] = loans_df['issue_d'].apply(pd.Timestamp)
    loans_df['term'] = loans_df['term'].apply(drop_months).astype(int)
    loans_df['emp_title'] = loans_df['emp_title'].apply(fix_emp)
    loans_df['term_end'] = loans_df['issue_d'] + pd.to_timedelta(loans_df['term'], unit = "D")
    return loans_df


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def correlations(loans_df, col_names):
    result = pd.Series()
    for i in np.arange(len(col_names)):
        testing_df = pd.DataFrame((loans_df[col_names[i][0]]))
        testing_df[col_names[i][1]] = loans_df[col_names[i][1]]
        r_stat = testing_df[col_names[i][0]].corr(testing_df[col_names[i][1]])
        row_name = "r_" + col_names[i][0] + "_" + col_names[i][1]
        result[row_name] = r_stat
    return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans_df):
    
    loans_df_copy = loans_df.copy()
    
    def assign_bin(fico_score):
        if 580 <= fico_score < 670:
            return "[580, 670)"
        elif 670 <= fico_score < 740:
            return "[670, 740)"
        elif 740 <= fico_score < 800:
            return "[740, 800)"
        elif 800 <= fico_score < 850:
            return "[800, 850)"
    
    loans_df_copy['bins'] = loans_df_copy['fico_range_low'].apply(assign_bin)
    
    fig = px.box(loans_df_copy, x='bins', y='int_rate', color='term',
                labels = {'int_rate': "Interest Rate (%)",
                         'bins': 'Credit Score Range',
                         'term': "Loan Length (Months)"},
                title = "Interest Rate vs. Credit Score",
                color_discrete_map = {'36': "#1F77B4",
                                     '60': '#FF7F0E'})
    
    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans_df, N):
    loans_copy = loans_df.copy()
    
    with_ps = loans_df[loans_df['desc'].notna()]['int_rate'].mean()
    without_ps = loans_df[loans_df['desc'].isna()]['int_rate'].mean()
    obs_diff = with_ps - without_ps
    
    diff = []
    for _ in range(N):
        with_shuffled = loans_copy.assign(shuff_desc=np.random.permutation(loans_copy['desc']))
        shuff_with_ps = with_shuffled[with_shuffled['shuff_desc'].notna()]['int_rate'].mean()
        shuff_without_ps = with_shuffled[with_shuffled['shuff_desc'].isna()]['int_rate'].mean()
        diff_mean = shuff_with_ps - shuff_without_ps
        diff.append(diff_mean)
    return (np.array(diff) > obs_diff).mean()
    
def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    return ("The p value for this test is small, so there is reason to reject the null hypothesis in favor the the alternative; " +
            "therefore, under the assumption that loanees are aware of interest rates are larger with PS they would deliberatly" +
             " not include PS which makes it NMAR")


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income_val, bracket_list):
    total_taxed = 0
    for i in np.arange(len(bracket_list)-1, -1, -1):
        if income_val > bracket_list[i][1]:
            taxed_val = income_val - bracket_list[i][1]
            income_val -= taxed_val
            taxed_val = taxed_val * bracket_list[i][0]
            total_taxed += taxed_val
    return total_taxed


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(tax_df):
    tax_copy = tax_df.copy()
    
    def remove_dollar(dollar_amount):
        dollar_amount = str(dollar_amount)
        dollar_amount = dollar_amount.replace("$", "").replace(",", "")
        dollar_amount = int(dollar_amount)
        return dollar_amount
    
    def change_rate(rate_amount):
        if rate_amount == "none":
            return 0.00
        else:
            rate_amount = float(str(rate_amount).strip("%")) / 100
            return round(rate_amount, 2)
        
    def drop_weird(state_name):
        state_name = str(state_name)
        if "(" in state_name:
            return np.NaN
        else:
            return state_name
    
    tax_copy = tax_copy.dropna(axis = 0, how='all')
    tax_copy['State'] = tax_copy['State'].apply(drop_weird)
    tax_copy['State'].replace('nan', np.NaN, inplace=True)
    tax_copy['State'] = tax_copy['State'].fillna(method='ffill')
    tax_copy['Lower Limit'] = tax_copy['Lower Limit'].fillna(0)
    tax_copy['Lower Limit'] = tax_copy['Lower Limit'].apply(remove_dollar)
    tax_copy['Rate'] = tax_copy['Rate'].fillna(0.00)
    tax_copy['Rate'] = tax_copy['Rate'].apply(change_rate)
    
    return tax_copy


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------

def state_brackets(taxes_df):
    taxes_copy = taxes_df.copy()
    taxes_copy['bracket_list'] = taxes_copy[['Rate', 'Lower Limit']].apply(tuple, axis = 1)
    taxes_copy = taxes_copy.drop(columns=['Rate', "Lower Limit"])
    taxes_copy = taxes_copy.groupby("State").agg({'bracket_list': lambda x: list(x)})
    return taxes_copy
    


def combine_loans_and_state_taxes(loans_df, taxes_df):
    import json
    loans_copy = loans_df.copy()
    state_brackets_df = state_brackets(taxes_df).reset_index()
    
    state_mapping = Path('data') / 'state_mapping.json'
    state_mapping = json.loads(state_mapping.read_text())
    
    state_brackets_df = state_brackets_df.replace({'State': state_mapping})
    
    results = loans_copy.merge(state_brackets_df, left_on='addr_state', right_on = 'State')
    results = results.drop(columns = ['addr_state'])
    return results


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loan_tax_df):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    
    tax_df = loan_tax_df.copy()
    tax_df['state_tax_owed'] = tax_df.apply(lambda x: tax_owed(x['annual_inc'], x['bracket_list']), axis = 1)
    tax_df['federal_tax_owed'] = tax_df.apply(lambda x: tax_owed(x['annual_inc'], FEDERAL_BRACKETS), axis = 1)
    tax_df['disposable_income'] = tax_df['annual_inc'] - tax_df['state_tax_owed'] - tax_df['federal_tax_owed']
    return tax_df


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------

def aggregate_and_combine(loans_df, professions, values, idx):
    result_df = pd.DataFrame(index = loans_df.groupby(idx).sum().index)
    for job in professions:
        result = (loans_df[loans_df["emp_title"].str.contains(job)].groupby(idx)['loan_amnt'].mean())
        kwargs = {f"{job}_mean_{values}" : result}
        result_df = result_df.assign(**kwargs)
        mean = loans_df[loans_df['emp_title'].str.contains(job)]['loan_amnt'].mean()
        result_df.loc["OVERALL", f"{job}_mean_{values}"] = mean
    result_df = result_df.dropna(axis = 0, how = 'all')
    return result_df

# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    result_df = aggregate_and_combine(loans, keywords, quantitative_column, categorical_column)
    first_cond = ((all(result_df.iloc[:-1, 0] > result_df.iloc[:-1, 1])) & (result_df.loc['OVERALL'][0] < result_df.loc['OVERALL'][1]))
    second_cond = ((all(result_df.iloc[:-1, 0] < result_df.iloc[:-1, 1])) & (result_df.loc['OVERALL'][0] > result_df.loc['OVERALL'][1]))
    return bool(first_cond | second_cond)
    
def paradox_example(loans):
    result = {
        'loans': loans,
        'keywords': ['manager', 'lead'],
        'quantitative_column': 'loan_amt',
        'categorical_column': 'verification_status'
    }
    return result
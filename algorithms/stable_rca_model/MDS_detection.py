import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency

def continuous_variable_mds_test(df_ref, df_test, columns_list, p_value_threshold=0.001):
    '''
    Given two input dataframes and a list of column names, output the columns with marginal distribution shift and test information dictionary.
    K-S test is employed for continuous data.
    '''
    shift_col_list = []
    ks_test_info = {}
    for col in columns_list:
        data1 = df_ref[col]
        data2 = df_test[col]

        statistic, p_value = ks_2samp(data1, data2)
        ks_test_info[col] = (statistic, p_value)
        if p_value < p_value_threshold:
            shift_col_list.append(col)

    return shift_col_list, ks_test_info

def categorical_variable_mds_test(df_ref, df_test, columns_list, p_value_threshold=0.001):
    '''
    Given two input dataframes and a list of column names, output the columns with marginal distribution shift and test information dictionary.
    chi2 test is employed for continuous data.
    '''
    shift_col_list = []
    chi2_test_info = {}
    for col in columns_list:
        counts1 = df_ref[col].value_counts()
        counts2 = df_test[col].value_counts()

        all_categories = set(counts1.index).union(set(counts2.index))
        counts1 = counts1.reindex(all_categories, fill_value=0)
        counts2 = counts2.reindex(all_categories, fill_value=0)

        contingency_table = pd.DataFrame([counts1, counts2], 
                                     index=["df_ref", "df_test"])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        chi2_test_info[col] = {'chi2': chi2, 'p_value': p_value, 'dof':dof, 'expected':expected}
        if p_value < p_value_threshold:
            shift_col_list.append(col)

    return shift_col_list, chi2_test_info 

class MDS_detector():
    def __init__(self, args):
        self.discrete_threshold = args.discrete_threshold
        self.p_value_threshold = args.p_value_threshold

    def separate_continuous_discrete(self, df):
        continuous_cols = []
        discrete_cols = []
        for column in df.columns:
            if np.issubdtype(df[column].dtype, np.number):
                # Numeric column
                unique_values = df[column].nunique()
                # If the number of unique value is smaller than or equal to the 
                # threshold multiplied by the total sample number, the variable 
                # is considered as discrete, otherwise it is considered as continuous.
                if unique_values <= int(self.discrete_threshold):
                    discrete_cols.append(column)
                else:
                    continuous_cols.append(column)
            else:
                # Non-numeric (categorical or string) column
                discrete_cols.append(column)

        return continuous_cols, discrete_cols
    
    def test(self, df_ref, df_test):
        self.continuous_cols, self.discrete_cols = self.separate_continuous_discrete(df_ref)
        shift_col_list_continuous, ks_test_info_continuous = continuous_variable_mds_test(df_ref, df_test, self.continuous_cols, self.p_value_threshold)
        shift_col_list_discrete, chi2_test_info_discrete = categorical_variable_mds_test(df_ref, df_test, self.discrete_cols, self.p_value_threshold)

        shift_col_list = shift_col_list_continuous + shift_col_list_discrete

        return shift_col_list
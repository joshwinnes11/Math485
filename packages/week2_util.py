""" 
Utility functions for working with Allegheny DHS Synthetic Data
    - @author Josh Winnes
    - PRECONDITIONS: various parameters
    - POSTCONDITIONS: various
    - PARAMETERS: various

"""

import pandas as pd
import numpy as np

def get_retention_ratio_matrix(df, recipient):
    
    df_retention = pd.merge(df, recipient[['id','first_date']], on = 'id', how = 'left')
    df_retention['elapsed'] = df_retention['date'].dt.month - df_retention['first_date'].dt.month
    df_retention_count = df_retention.groupby(["first_date", "elapsed"]).agg(
    active = ("id", "nunique"),
    ).reset_index()
    df_retention_count = df_retention_count.pivot(index = "first_date", columns="elapsed", values='active')
    df_retention_ratio = df_retention_count.reset_index()
    df_retention_ratio = df_retention_count.div(df_retention_ratio.iloc[:,1].to_numpy(),axis = 0)
    
    return df_retention_ratio


def get_correlation_matrix(df):
    
    df_temp = df.groupby(["id","serv"]).agg(
    num_serv = ('service', 'nunique') # this will be 1 or 0, "service" is categorical 
    ).reset_index()
    
    df_serv = df_temp.pivot_table(
    values='num_serv', 
    index=["id"], columns="serv", aggfunc=np.sum
    ).reset_index()
    
    correlation = df_serv.iloc[:,1:23].corr(method="spearman")
    
    return correlation


    
    
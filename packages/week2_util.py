""" 
Utility functions for working with Allegheny DHS Synthetic Data
    - @author Josh Winnes
    - PRECONDITIONS: various parameters
    - POSTCONDITIONS: various
    - PARAMETERS: various

"""

def get_sum_associated_servs(cohort_services):
    
    df_temp_cohort = cohort_services.groupby(["id","serv"]).agg(
    num_serv = ('service', 'nunique') # this will be 1 or 0, "service" is categorical 
    ).reset_index()

    df_serv_cohort = df_temp_cohort.pivot_table(
    values='num_serv', 
    index=["id"], columns="serv", aggfunc=np.sum
    ).reset_index()

    list_cohort_servs = pd.DataFrame(df_serv_cohort.iloc[:,1:22].sum())
    list_cohort_servs.columns = ['count']
    return list_cohort_servs
    


    
    
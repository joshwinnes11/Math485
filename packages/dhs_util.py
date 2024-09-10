import pandas as pd
import numpy as np

def dhs_preprocessing(df):

    ''' Data cleaning rules we follow in this work
        drop the columns that contain NA entirely
        drop the columns that has only one value in its entirety as they have not information. 
        
        because we do not know enough details about how the reocrd data are created/collected
        we will not remove records with age greater than a value, 
        nor will we remove records with seemingly conflict between age and education level
    ''' 
    df.dropna(axis=1, how='all', inplace=True) 
    df.drop(['synthetic_data', 'CALDR_YR'], axis=1, inplace=True) 

    ''' Data transformation we will apply for this work
        lower the case of the column names
        change some column names for ease of remembering 
        add a column for month name 
        change some columns into categorical type
    '''
    df.rename(columns=str.lower, inplace=True) 
    df.rename(columns = {'mci_uniq_id': 'id', 
                        'date_of_event': 'date', 
                        'marital_status': 'marital', 
                        'education_level':'education'}, inplace=True) 
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df["date"].dt.month_name()

    df = df.astype({'id':'category', 
                    'service':'category',
                    'age': 'int',
                    'gender':'category',
                    'race':'category',
                    'marital':'category',
                    'education':'category'})
    return df

def get_recipient_attribute(df,remove_unused_id = True):  # making the removal of unused ID to be default
    if remove_unused_id:    
        recipient = df.groupby(df['id'].cat.remove_unused_categories()).agg(
            num_service = ('service', 'count'),
            distinct_service = ('service', 'nunique'), 
            first_date = ('date', 'min'), 
            last_date = ('date', 'max'), 
            num_month = ('month', 'count'), 
            distinct_month = ('month', 'nunique')
        ).reset_index()
    else:
        recipient = df.groupby(['id']).agg(
            num_service = ('service', 'count'),
            distinct_service = ('service', 'nunique'), 
            first_date = ('date', 'min'), 
            last_date = ('date', 'max'), 
            num_month = ('month', 'count'), 
            distinct_month = ('month', 'nunique')
        ).reset_index()
    # do not include the columns that were keys to the groups for aggregation, 'service', 'date', 'month' in this case
    recipient = pd.merge(recipient, df.drop(['service','date','month'],axis=1), on='id', how='left')
    recipient = recipient.drop_duplicates(subset=['id'])
    
    return recipient

def get_recipient_month_attribute(df):
    recipient_month = df.groupby(['id', 'date']).agg(
        num_service = ('service', 'count'),
    ).reset_index()
    recipient_month = pd.merge(recipient_month, df.drop(['service','date','month'],axis=1), on='id', how='left')
    recipient_month = recipient_month.drop_duplicates(subset=['id', 'date'])

    return recipient_month

def add_service_label(df):
    serv_list =  ['S'+str(i).zfill(2) for i in range(1,1+df.service.unique().shape[0])] # create list of S01, S02, ..., S22
    service_map = dict(zip(df.service.unique(), serv_list)) # create a dictionary of service names to S01, S02, ..., S22
    df['serv'] = df['service'].map(service_map) #adding a new column with the service label
    return df, service_map # service_map need to be returned if we ever want to do a reverse lookup

def service_lookup(serv, service_map): # look up the service name by service label S01, S02, etc.
    return next(key for key, value in service_map.items() if value == serv)

def add_age_bin(df):
    bins = np.linspace(0, 100, 11).astype(int) # create bins
    bins_lower = np.roll (np.linspace(0, 100, 11).astype(int) -1, -1 ) # get the value of one lower of the bin max

    bin_names = [ a +'-' + b for a,b in zip(bins.astype(str), bins_lower.astype(str))] # put the two ends of the bin goether as bin names
    bin_names[-1]=bins.astype(str)[-1] + '+' # modify the bin name for the last bin
    bin_name_map = dict(zip(range(1, 1+len(bin_names)), bin_names)) # mkae the dictionary for mapping
    df['age_bin'] = np.digitize(df.age, bins) # put actual age into bins
    df["age_bin"] = df["age_bin"].map(bin_name_map).astype('category') # map the age bin number into the bin name
    return df
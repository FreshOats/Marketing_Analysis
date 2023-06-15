### Wash_Data 
''' 
The function Wash_Data() opens a csv file from the provided path and cleans the 'Leads.csv' data with 3 subroutines:
    1. Open_File(), which opens the csv from the specified path returning the native data as a dataframe
    2. Scrub_Columns(), which removes the unnecessary columns determined in the EDA
    3. Mop_Data(), which cleans up some discrepancies and changes some data to nan values
'''

import pandas as pd
import numpy as np


def Open_File(path):
    # import pandas as pd
    df = pd.read_csv(path)
    return df

def Scrub_Columns(df): 
    df.drop(columns=['Magazine',
                     'Receive More Updates About Our Courses',
                     'Update me on Supply Chain Content',
                     'Get updates on DM Content',
                     'I agree to pay the amount through cheque',
                     'What matters most to you in choosing a course',
                     'City'
                     ], inplace=True)
    return df

def Mop_Data(df): 
    # import numpy as np

    # Define the dictionaries needed for the changes
    sources = {'google': 'Google',
               'welearnblog_Home': 'WeLearn'
               }
    
    occupations = {
        'Working Professional': 'Employed',
        'Other': np.nan,
        'Housewife': 'Unemployed',
        'Businessman': 'Employed'
        }

    levels = {
        '02.Medium': 'Medium',
        '01.High': 'High',
        '03.Low': 'Low'
        }
    
    # Fix the Lead Source issues
    df['Lead Source'].replace(sources, inplace=True)
    
    # Change Select to nan in the 3 affected columns
    df['Specialization'].replace({'Select': np.nan}, inplace=True)
    df['How did you hear about X Education'].replace(
        {'Select': np.nan}, inplace=True) 
    df['Lead Profile'].replace(
        {'Select': np.nan}, inplace=True)
    
    # Fix Current Occupations
    df['What is your current occupation'].replace(occupations, inplace=True)

    # Fix Asymmetric Indexes
    df['Asymmetrique Activity Index'].replace(levels, inplace=True)
    df['Asymmetrique Profile Index'].replace(levels, inplace=True)

    return df

def Wash_Data(path):
    df = Open_File(path)
    df = Scrub_Columns(df)
    df = Mop_Data(df)
    return df
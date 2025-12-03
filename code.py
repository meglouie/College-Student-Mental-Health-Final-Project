'''
Name: Megan Louie
Date: 12/1/2025
Description: code that analyzes the data from HMS from 2017-2025
'''

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# MAIN
################################################################################

def main():
    # Reading in the data
    file_2017 = read_csv('data/HMS_2017-2018_PUBLIC_instchars.csv')
    #print(file_2017)
    cleaned_2017 = clean_data(file_2017)
    cleaned_2017 = combine_columns(cleaned_2017, 'drug_use', ['drug_none', 'drug_mar', 'drug_coc'])
    cleaned_2017 = combine_columns(cleaned_2017, 'substance_use', ['alc_any', 'drug_use'])
    cleaned_2017 = combine_columns(cleaned_2017, 'abuse_history', ['abuse_life', 'substance_use'])
    cleaned_2017 = combine_columns(cleaned_2017, 'mental_health', ['dx_dep1', 'sui_idea'])
    cleaned_2017 = combine_columns (cleaned_2017, 'anxiety', ['dx_anx', 'anx_score'])
    cleaned_2017 = combine_columns(cleaned_2017, 'risk_factors', ['abuse_history', 'mental_health'])
    # Define the columns we want in the final cleaned DataFrame
    desired_cols = ['gender', 'age', 'drug_use', 'substance_use',
                    'abuse_history', 'mental_health', 'risk_factors', 'anxiety']

    # select_columns will pick available columns and warn about missing ones
    selected = select_columns(cleaned_2017, desired_cols)

    # Create df_cleaned_2017 from the selected columns (copy to avoid chained-assignment issues)
    df_cleaned_2017 = selected.copy()      

    # Basic post-processing: reset index and ensure consistent column order
    df_cleaned_2017 = df_cleaned_2017.reindex(columns=[c for c in desired_cols if c in df_cleaned_2017.columns])
    df_cleaned_2017.reset_index(drop=True, inplace=True)

    # Save the full cleaned DataFrame to CSV so you can inspect all rows
    out_path = 'data/df_cleaned_2017.csv'
    try:
        df_cleaned_2017.to_csv(out_path, index=False)
        print(f"Saved cleaned DataFrame to: {out_path}")
    except Exception as e:
        print(f"Warning: could not save cleaned DataFrame to {out_path}: {e}")

    #calculate stats for each column
    for column in df_cleaned_2017.columns:
        mean = calc_mean(df_cleaned_2017, column)
        value_range = calc_range(df_cleaned_2017, column)
        std_dev = calc_std(df_cleaned_2017, column)
        min_value = calc_min(df_cleaned_2017, column)
        max_value = calc_max(df_cleaned_2017, column)
        print(f"Column: {column}")
        print(f" Mean: {mean}")
        print(f" Range: {value_range}")
        print(f" Standard Deviation: {std_dev}")
        print(f" Minimum: {min_value}")
        print(f" Maximum: {max_value}")
    
    
################################################################################
# HELPER FUNCTIONS
################################################################################

def read_csv(filename):
    '''
    reading in the csv files
    '''
    file = pd.read_csv(filename, sep=',', on_bad_lines='skip', encoding="latin1")
    return file

def clean_data(data):
    '''
    cleaning the data
    '''
    # drop rows that are entirely NA (but keep rows with some values)
    cleaned_data = data.dropna(how='all').copy()

    # coerce age to numeric where available (invalid parsing becomes NaN)
    if 'age' in cleaned_data.columns:
        cleaned_data['age'] = pd.to_numeric(cleaned_data['age'], errors='coerce')

    # cleaning N/A values
    cleaned_data.replace(['N/A', 'NaN', '', ' '], np.nan, inplace=True)
    #clenaing outliers
    if 'age' in cleaned_data.columns:
        cleaned_data = cleaned_data[(cleaned_data['age'] >= 17) & (cleaned_data['age'] <= 50)]
    return cleaned_data

def select_columns(data, columns):
    '''
    selecting specific columns from the data
    '''
    # Be tolerant: if some columns are missing, select the intersection and warn
    available = [c for c in columns if c in data.columns]
    missing = [c for c in columns if c not in data.columns]
    if len(missing) > 0:
        print(f"Warning: the following requested columns are missing and will be skipped: {missing}")
    selected_data = data[available].copy()
    return selected_data

def combine_columns(data, new_column_name, columns_to_combine):
    '''
    combining multiple columns into one
    '''
    # Only combine columns that actually exist in the DataFrame
    existing = [c for c in columns_to_combine if c in data.columns]
    if not existing:
        # if none of the requested columns exist, create the column with NaN
        data[new_column_name] = np.nan
        return data

    # Convert to string and replace NaN with empty string so join works
    # Then drop any empty parts when joining
    def _join_row(row):
        parts = [str(x).strip() for x in row.values if pd.notna(x) and str(x).strip() != '']
        return '_'.join(parts) if parts else np.nan

    data[new_column_name] = data[existing].apply(_join_row, axis=1)
    return data

def calc_mean(data, column):
    '''
    calculating the mean of a column
    '''
    if column in data.columns:
        mean_value = data[column].mean()
        return mean_value
    else:
        return None
def calc_range(data, column):
    '''
    calculating the range of a column
    '''
    if column in data.columns:
        min_value = data[column].min()
        max_value = data[column].max()
        return (min_value, max_value)
    else:
        return None
    
def calc_std(data, column):
    '''
    calculating the standard deviation of a column
    '''
    if column in data.columns:
        std_value = data[column].std()
        return std_value
    else:
        return None
    
def calc_min(data, column):
    '''
    calculating the minimum of a column
    '''
    if column in data.columns:
        min_value = data[column].min()
        return min_value
    else:
        return None
    
def calc_max(data, column):
    '''
    calculating the maximum of a column
    '''
    if column in data.columns:
        max_value = data[column].max()
        return max_value
    else:
        return None

if __name__ == "__main__":
    main() 
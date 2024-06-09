import pandas as pd

final_updated_data = pd.read_csv("final_data_updated.tsv", sep='\t')
asd_data = pd.read_csv("asd_data.csv",sep=',', dtype={'asd': str})


final_updated_data['target'] = ''

for index, row in final_updated_data.iterrows():
    sample_name = row['SampleName']  

    matching_row = asd_data.loc[asd_data['subject_sp_id'] == sample_name]

    if not matching_row.empty:  
        asd_value = matching_row.iloc[0]['asd']
        if asd_value == 'True':
            final_updated_data.loc[index, 'target'] = 1
        if asd_value == 'False':
            final_updated_data.loc[index, 'target'] = 0
    else:
        print(f"No matching sample name found in ASD data for {sample_name}")  

final_updated_data.to_csv("final_updated_data_with_target.tsv", sep='\t', index=False)


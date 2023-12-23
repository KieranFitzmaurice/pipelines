import numpy as np
import os
import requests

pwd = os.getcwd()
data_dir = os.path.join(pwd,'data')
bucket_url = 'https://dfo-test-2.s3.amazonaws.com'
keys_url = bucket_url + '/keys.txt'

keys = requests.get(keys_url).text.splitlines()
experiments = np.unique(['/'.join(x.split('/')[:-1]) for x in keys])
experiments = experiments[::-1]

for i,experiment in enumerate(experiments):
    
    print(f'Experiment {i+1}/{len(experiments)}: {experiment}')
    
    experiment_keys = [x for x in keys if x.startswith(experiment)]
    output_dir = os.path.join(data_dir,f'DAS_data/DAS/Raw/{experiment}')
    os.makedirs(output_dir,exist_ok=True)

    for key in experiment_keys:
        
        file_url = bucket_url + f"/{key}"
        file_name = key.split('/')[-1]
        file_path = os.path.join(output_dir,file_name)
        res = requests.get(file_url)

        with open(file_path,'wb') as f:
            f.write(res.content)
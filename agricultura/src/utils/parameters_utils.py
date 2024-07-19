import yaml
import os

def write(base_folder = ''):
    # namelist = os.listdir('pipelines_dummy_dataset/dataset')
    if base_folder=='':
        path = 'pipelines_dummy_dataset/dataset'
        params_file_name = 'params.yaml'
    else:
        path = base_folder + '/' + 'pipelines_dummy_dataset/dataset'
        params_file_name = base_folder + '/' + 'params.yaml'

    full_namelist = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    #check and clean namelist
    namelist = []
    for element in full_namelist:
        if os.path.exists(path + '/' + element + '/abaqus.dvc'):
            namelist.append(element)
    
    dicti = dict.fromkeys(namelist)

    with open(params_file_name) as file:
        params = yaml.safe_load(file)

    params['collections']=dicti
    print(params)
    with open(params_file_name, 'w') as file:
        yaml.safe_dump(params, file)


def create_dvc_pipeline():
    directory = os.environ['DATA_SUBDIRECTORY']
    with open("params.yaml") as f:
        p = yaml.safe_load(f)
        p['dataset']['images_path'] = f"dataset/{directory}"
        p['predict']['results_dir'] = f"results/{directory}"
        print("The configuration parameters are set to :" + yaml.dump(p, default_flow_style=False, sort_keys=False))
    with open("dvc.yaml") as f:
        d = yaml.safe_load(f)
        d['stages']['predict']['outs'][0] = f"results/{directory}/cytotoxicity"
        d['stages']['predict']['outs'][1] = f"results/{directory}/mk"
        d['stages']['create_report']['deps'][0] = f"results/{directory}/cytotoxicity"
        d['stages']['create_report']['deps'][1] = f"results/{directory}/mk"
        d['stages']['create_report']['outs'][0] = f"results/{directory}/statistical_reports"
        d['stages']['upload_results']['deps'][0] = f"results/{directory}/statistical_reports"
        print(yaml.dump(d, default_flow_style=False, sort_keys=False))
    with open("params.yaml", "w") as fw:
        yaml.dump(p, fw, default_flow_style=False, sort_keys=False)
    with open("dvc.yaml", "w") as fw:
        yaml.dump(d, fw, default_flow_style=False, sort_keys=False)

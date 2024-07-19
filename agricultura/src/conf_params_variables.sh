#!/usr/bin/python3
import yaml
import os
directory = os.environ['RESULTS_DIRECTORY']

with open("params.yaml") as f:
    p = yaml.safe_load(f)
    p['dataset']['images_path'] = f"dataset/{directory}"
    p['predict']['results_dir'] = f"results/{directory}"
    print("The configuration parameters are set to :" + yaml.dump(p, default_flow_style=False, sort_keys=False))


with open("dvc.yaml") as f:
    d = yaml.safe_load(f)
    d['stages']['predict']['outs'][0] = f"results/{directory}"
    d['stages']['upload_results']['deps'][0] = f"results/{directory}"
    print(yaml.dump(d, default_flow_style=False, sort_keys=False))

with open("params.yaml", "w") as fw:
    yaml.dump(p, fw, default_flow_style=False, sort_keys=False)

with open("dvc.yaml", "w") as fw:
    yaml.dump(d, fw, default_flow_style=False, sort_keys=False)

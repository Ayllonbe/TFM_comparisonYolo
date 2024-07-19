
# Yolov5 Object Detection Model Training 

The template is designed for and tested to work with the [Yolo V5](https://github.com/ultralytics/yolov5) 
object detection model on the input tiled data. [📄 One-Pager](https://jisap.basf.net/api/v1/projects/14767/files/Yolov5TemplateOnepager%2Epdf?ref=documentation) 
[📕 Documentation PDF](https://jisap.basf.net/api/v1/templates/14767/files/manual.pdf?ref=master) [🌍Documentation HTML](https://jisap.basf.net/api/v1/templates/14767/files/manual.html?ref=master)

- Processes training data (images + bounding box labels in DetectNet format as text files)
- Trains yolov5 model with given parameters (params.yaml) and produces a training report
- Makes model available through an JISAP Release (it could be deployed to API)
✓ Best-practice blueprint to organize training datasets, model parameters, training code,  execution, metrics report, model artifacts, and automated pipeline deployment.


## How to use the template? 
-The templates provide the necessary source code and DVC pipeline to train an object detector.To apply the template, from the existing JISAP project you can click into the selection section. When you apply a template, a Gitlab merge request is created.

<img src="https://jisap.basf.net/api/v1/projects/14767/files/images/ApplyTemplate%2Epng?ref=master" width="800">

-All aspects can be modified, extended, removed, as applying a template copies-in the template code into your working-project, free to be modified as needed.

## Template Quick Overview
The aim opf the template is to give a blueprint to train a object detection model (Yolo V5 library) that detects the classes  of structures, defined in dataset.yaml. Further details in the documentation can be found in the documentation submodule [README](https://gitlab.roqs.basf.net/jisap/documentation/object-detection-template-documentation/-/blob/documentation/index.md)

<img src="https://jisap.basf.net/api/v1/projects/14767/files/images/TemplatePipeline%2Esvg?ref=master" width="800">

### Development

The pipeline mainly specified in these files:

| File                                 |Content    |
|------                                |---------  |
| [dvc.yaml](dvc.yaml)                 | Reproducible execution pipeline for model training and/or inference. See DVC [docu](https://dvc.org/doc/start) |
| [params.yaml](params.yaml)           | Settings and parameters for data processing and yolov5 model training and inference|
| [requirements.txt](requirements.txt) | List of python packages, needed for training and deployment |
| [src](src) | Folder Conatining all the source code to complete each step of the DVC Pipeline|
| [.gitlab-ci.yml](.gitlab-ci.yml)     | Gitlab CI/CD automation, server-based execution of the workflow. Parts of the stages are defined in common template files. For the reproduce step, the before_script and after_script steps are defined in [reproduce-step-template.yml](https://gitlab.roqs.basf.net/jisap/backend-services/appstoredeployment/-/blob/master/reproduce-step-template.yml). Certain variables may be overwritten.|

####  CI/CD Workflow

For running the pipeline some necessary CI/CD variables needo to be defined in Settings Tab:
- __repo_token__ : Personal access token to use as credentials [Token creation](https://gitlab.roqs.basf.net/-/profile/personal_access_tokens)

#### Local Execution

**Requirements**
The local exection of this pipeline should be done in Linux and python>=3.8. 
All the python packages are listed under requirements.txt
Besides, [jisap-cli](https://gitlab.roqs.basf.net/ktc/jisap-cli) and [DVC](https://dvc.org/doc/install/linux) dependencies need to be installed. 
For local development of the code and running the code:

1. Clone source code: `git clone [URL]` and set the project directory `cd [project directory]` 
2. Create an [virtual environment](https://virtualenv.pypa.io/en/latest/) with a suitable python version: `virtualenv NAME -p /usr/bin/python3.9`. Then activate the environment (`source NAME/bin/activate`) and then install the python requirements: `python3 -m pip install --default-timeout=1000 -r requirements.txt`.
3. Configure DVC remote to allow dataset download as well as result file upload: `dvc remote modify --local jisap-basf password ` [[ACCESS_TOKEN]](https://gitlab.roqs.basf.net/-/profile/personal_access_tokens) 
4. Download DVC-managed binary files: `dvc pull --jobs 4 -R`


## Authors

*This template has been created by Maria Monzon and Christian Klukas.*

In case of questions, contact the authors by mail or post your question in the [JISAP Teams Channel](https://teams.microsoft.com/l/channel/19%3aTUWHCbIw5JQXSuXJbQkIMw5L590GFi1Aylo5P1KY-ns1%40thread.tacv2/Allgemein?groupId=6ca75e12-2507-4b0f-8517-0d8fb1cc740d&tenantId=ecaa386b-c8df-4ce0-ad01-740cbdb5ba55).

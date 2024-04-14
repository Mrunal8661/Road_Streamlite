# Road Damage Detection Applications

This project is road damage detection applications that designed to enhance road safety and infrastructure maintenance by swiftly identifying and categorizing various forms of road damage, such as potholes and cracks.

## Performing Detection Using Image
![Detection using image](resource/RDD_Image_Example.gif)

## Performing Detection Using Video
![Detection using video](resource/RDD_Video_Example.gif)

The project is powered by YOLOv8 deep learning model that trained on Crowdsensing-based Road Damage Detection Challenge 2022 dataset.

There is four types of damage that this model can detects such as:
- Longitudinal Crack
- Transverse Crack
- Alligator Crack
- Potholes

## Running on Local Server

This is the step that you take to install and run the web-application on the local server.

``` bash
# Install CUDA if available
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

# Create the python environment
conda create -n rdd python=3.8
conda activate rdd

# Install pytorch-CUDA
# https://pytorch.org/get-started/locally/
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install ultralytics deep learning framework
# https://docs.ultralytics.com/quickstart/
pip install ultralytics

# Clone the repository
git clone https://github.com/oracl4/RoadDamageDetection.git
cd RoadDamageDetection

# Install requirements
pip install -r requirements.txt

# Start the streamlit webserver
streamlit run Home.py
```

/home/oracl4/project/rdd/dataset/RDD2022/
├── RDD2022_all_countries
│   ├── China_Drone
│   │   └── train
│   │       ├── annotations
│   │       │   └── xmls
│   │       ├── images
│   │       └── labels # Created after prepare dataset process
│   ├── China_MotorBike
│   │   └── ...
│   ├── Czech
│   │   └── ...
│   ├── India
│   │   └── ...
│   ├── Japan
│   │   └── ...
│   ├── Norway
│   │   └── ...
│   └── United_States
│       └── ...
└── rawData # Not Used, .zip folder
```

Perform the dataset conversion from PascalVOC to YOLOv8 format using **0_PrepareDatasetYOLOv8.ipnb** notebook. This will also create a train and val split for the dataset due to lack of test labels on the original dataset. It will also remove excess background image from the dataset. It will copy the dataset and create a new directory on the training folder.

```
├── dataset
│   └── rddJapanIndiaFiltered
│       ├── India
│       │   ├── images
│       │   │   ├── train
│       │   │   └── val
│       │   └── labels
│       │       ├── train
│       │       └── val
│       ├── Japan
│       │   └── ...
│       └── rdd_JapanIndia.yaml # Create this file for YOLO dataset config
└── runs
```

Run the training on **1_TrainingYOLOv8.ipynb** notebook. You can change the hyperparamter and training configuration on that notebook.

## Evaluation Result

This is the training result of the YOLOv8s model that trained on the filtered Japan and India dataset with RTX2060 GPU. You can perform the evaluation on your dataset with **2_EvaluationTesting.ipynb** notebook, just convert your dataset into ultralytics format.

<p align="center">
    <img src='resource/PR_curve.png' width='80%'>
    <img src='resource/confusion_matrix.png' width='80%'>
    <img src='resource/val_batch2_pred.jpg' width='100%'>
</p>


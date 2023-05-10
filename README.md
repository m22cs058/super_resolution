# Image Super-Resolution using GANs

This repository implements an image super-resolution algorithm based on the SRGAN architecture. The goal of this project is to enhance the resolution of low-resolution images by a factor of 4, resulting in visually sharper and more detailed images.
<p  align="center">
<img  width="350"  src="https://github.com/m22cs058/super_resolution/blob/main/samples/test_lr.jpg?raw=true"  alt="Material Bread logo">
<img  width="350"  src="https://github.com/m22cs058/super_resolution/blob/main/samples/test_hr.jpg?raw=true"  alt="Material Bread logo">
</p>

## Dataset
The DIV2K_HR dataset was used to train our SRGAN (Super-Resolution Generative Adversarial Network) model. DIV2K_HR is a popular dataset widely used in the field of image super-resolution research. It consists of high-resolution images collected from various sources, such as the internet, professional photography, and digital cameras. The dataset contains a diverse range of subjects, including natural landscapes, objects, animals, and people.
To download the dataset, move to the project directory and run the following:

    !wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
    !unzip DIV2K_train_HR.zip

## Setup

Create a conda environment

    conda create -n super_res python=3.9.13
    conda activate super_res
    
Change directory to project folder and install requirements

    cd super_resolution
    pip install -r requirements.txt
To train the model

    python train.py
To inference on the model

    streamlit run app.py

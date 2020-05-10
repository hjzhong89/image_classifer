# Intro to Machine Learning - TensorFlow Project

This project was done as a part of Udacity's Intro to Machine Learning course. In this directory you will find the 
project's completed Jupyter notebook as well as the HTML export. Included is a CLI tool for running trained models to 
classify an image and display the results.

## Usage

1. Create an environment and install the required packages using `pip`
```
python3 -m venv env
source venv/bin/activate
pip install -r requirements.txt
```
1. Basic usage
```
python predict.py test_images/wild_pansy.jpg feed_forward_model_1588920328.h5
```
1. See options
```
pythong predict.py -h
```
# Image Classification

## Overview

This project focuses on image classification using a Kaggle dataset containing images and their corresponding weather status classifications. The tasks include exploratory data analysis, development, and evaluation of a multi-class classification model to detect weather conditions in images, and creating an API to serve the trained model.

## Tasks

### 1. Exploratory Data Analysis (EDA)
- Conducted an exploratory data analysis on the Kaggle dataset.
- Analyzed the distribution of weather classes.
- Visualized sample images from each weather condition.

### 2. Model Development and Evaluation
- Developed a multi-class classification model using a machine learning framework (TensorFlow).
- Split the dataset into training and testing sets.
- trained the model on the training set and evaluate its performance on the testing set.
- Generated plots for accuracy, test accuracy, and cost.

### 3. API Development
- Created an API using Flask framework.
- Implementd one endpoint that accepts an image (PNG or JPG only).
- There some problem with the response that I wasn't able to detect.

### 4. Dockerization
- Provided a Dockerfile to build a Docker image for the API.
- I faced some problems however because the Docker isn't downloading numpy
## Getting Started

### Prerequisites
- Python 3.x
- Preferred machine learning framework installed (TensorFlow)
- Preferred web framework installed (Flask)
- Docker installed (for building and running Docker images)

### File Structure
- **data/:** from the Kaggle dataset
- **weather_image_classifier_notebook/:**  Jupyter notebook (it has the data preprocessing/ EDA, model training, and prediction functions).
- **api/:** Directory for the API implementation.
  - **app.py:** Main API script.
  - **requirements.txt:** File specifying required Python packages.
  - **model.py:** File to use the parameters to classify an input image.
  - **Dockerfile:** File for building a Docker image.
- **README.md:** Project documentation.
- **saved_variables.json:** parameters of the trained model. 
- **index.html:** interface for the API
## Usage

### Exploratory Data Analysis:
- Run notebooks in the notebooks/ directory for exploratory data analysis.


### API Development:
- Navigate to the api/ directory.
- Install required packages: `pip install -r requirements.txt`
- Run the API: `python app.py`

### Dockerization:
- Build Docker image: `docker build -t image_classification_api .`
- Run Docker container: `docker run -p 5000:5000 image_classification_api`

## API Endpoint

- **Endpoint:** /classify
- **Method:** POST
- **Request:** Send an image file (PNG or JPG) as part of the request.
- **Response:** JSON object containing the weather condition classification result.

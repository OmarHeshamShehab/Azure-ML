# Temperature Conversion Using Linear Regression on Azure Machine Learning

This project contains three Jupyter notebooks that demonstrate how to use Azure Machine Learning to build, train, and deploy a linear regression model that converts temperatures between Celsius and Fahrenheit. The notebooks walk through the process of setting up the Azure environment, training the model, and submitting experiments to Azure's cloud infrastructure.

## Table of Contents
- [Project Overview](#project-overview)
- [Notebooks Description](#notebooks-description)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

The purpose of this project is to demonstrate how to use Azure Machine Learning to train and deploy machine learning models. Specifically, a linear regression model is trained to convert Celsius to Fahrenheit temperatures. The project is structured into three main Jupyter notebooks, each covering a different part of the process. The notebooks also showcase how to submit machine learning experiments to Azure for training in a cloud environment.

## Notebooks Description

### 1. `test.ipynb`
   - **Description**: This notebook sets up an Azure Machine Learning workspace, defines an environment with the necessary dependencies (`scikit-learn`), and trains a linear regression model to convert Celsius to Fahrenheit. The trained model is saved as `model.pkl` and the experiment is submitted to Azure for execution.
   - **Key Features**:
     - Azure ML workspace setup.
     - Linear regression for temperature conversion.
     - Model is saved using `joblib`.

### 2. `Celsius-Fahrenheit.ipynb`
   - **Description**: This notebook also trains a linear regression model for Celsius to Fahrenheit conversion. It saves the trained model as `model.h5`, likely using TensorFlow or Keras. The notebook includes steps for setting up and submitting the experiment to Azure's cloud infrastructure.
   - **Key Features**:
     - Linear regression for temperature conversion.
     - Model is saved using `model.h5`.
     - Azure ML workspace setup and experiment submission.

### 3. `L-R.ipynb`
   - **Description**: This notebook follows the same pattern as the others, focusing on training a linear regression model for temperature conversion. It uses Azure Machine Learning to submit the experiment, with the environment and dependencies handled via `Conda`.
   - **Key Features**:
     - Linear regression for temperature conversion.
     - Azure ML workspace setup and submission.
     - Model is saved using `joblib`.

## Setup and Installation

To run the notebooks in this project, you'll need the following:

### Prerequisites
- Python 3.x installed.
- Jupyter installed (`pip install jupyterlab`).
- Azure subscription with Machine Learning workspace set up.
- Azure ML SDK installed (`pip install azureml-sdk`).

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/username/project-name.git
    ```

2. Navigate to the project directory:
    ```bash
    cd project-name
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up your Azure Machine Learning workspace by following the instructions in the notebooks.

## Usage

1. Open the Jupyter notebooks in JupyterLab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```

2. Follow the steps in each notebook:
    - For `test.ipynb`: Train and save a model for Celsius to Fahrenheit conversion.
    - For `Celsius-Fahrenheit.ipynb`: Train and save the model in `model.h5`.
    - For `L-R.ipynb`: Train the model and submit the experiment to Azure.

3. Ensure your Azure credentials are correctly configured in each notebook.

## The project relies on the following libraries:
- `scikit-learn`
- `joblib`
- `Azure ML SDK`

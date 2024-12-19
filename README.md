## Kaggle Housing Issue 
This repository does not aim to provide the best or simplest solution to the housing problem. Instead, it represents my personal approach, using the tools and libraries I prefer. Specifically, I used PyTorch for modeling and Scikit-learn for tasks like normalization and recursive feature elimination (RFE).

While TensorFlow might require less code due to its built-in normalization layers, I chose PyTorch to explore its capabilities in a regression problem, even though it may not be the most straightforward choice for such tasks. This project is more about experimenting and seeing where this approach leads.

(Note: I work with standard Python scripts and leverage the Jupyter Interactive Window for development and debugging. It’s a convenient and efficient way to test code interactively, and I highly recommend it for projects like this.)

PyTorch is used with CUDA and cuDNN acceleration, make sure to adjust the environment and configurations properly if using the libraries directly from the `requirements.txt` file. Direct installation from `requirements.txt` may not work as expected due to dependencies on GPU acceleration.

Since the housing problem is relatively straightforward, I won’t delve into excessive detail here. Below, you can find the project structure and file descriptions. Additionally, the scripts are well-documented for better understanding. (Note: A GAN experiment will be added soon.)
## Duplicating the .env File
To download the housing dataset using the Kaggle API, you need to duplicate the `.env.example` file and rename it to `.env` (or simply rename `.env.example` to `.env`, whichever you prefer). Then, add your credentials as shown in the `.env.example` file.

If you prefer not to use the API, you can manually download the dataset from [here](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) and place it in the `data/raw` folder.
## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── data
    │   ├── dataset_download.py <- Code to download the dataset from Kaggle using the API      
    │   └── process_data.py     <- Code to preprocess the dataset
    |
    ├── modeling      
    |   ├── models              <- Trained model weights          
    │   ├── __init__.py 
    │   ├── GAN_train.py        <- Script for training a generative model to create synthetic data
    │   ├── MLR_train.py        <- Script for training a Multi-Linear Regression model
    |   ├── PLR_train.py        <- Script for training a Polynomial Regression model
    |   └── predict.py          <- Code to run model inference with trained models 
    │
    ├── services                
    │   └── __init__.py         <- Service classes to connect with external platforms, tools, or APIs
    │
    ├── visualization 
    |   └── EDA.py              <- Plotting for Exploratory Data Analysis 
    |
    └── utility
        └── plot_settings.py    <- My rcParams plot styling settings
```

--------

## Project Template
If you liked the project structure, you can check out my template repository:

[![GitHub Repo](https://img.shields.io/badge/GitHub-Project%20Template-black?style=for-the-badge&logo=github)](https://github.com/yigitcanozdemir/data-science-project-template)

# Disaster Response Pipeline Project

### Description

This Project is part of Udacity's Data Science Nanodegree Program. The dataset is provided by Figure Eight and it contains pre-labelled tweet and messages from real-life disaster. The goal of this project is to create a Natural Language Processing tool that can categorize messages.

The Project consists of 3 sections:

1. ETL (Extract, Load, Transform) Pipeline which imports data, cleans it and saves it in the database structure using SQLite.
2. Machine Learning Pipeline which trains the model and improves it using grid search for hyper parameter tuning.
3. Web App that takes an input message and provides classification results along with data visualizations.

### Installation

Clone this GIT repository:

```
https://github.com/Aleksandra-Rojek-p/disaster_response.git
```

### Instructions to execute the program
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Code files

- `app/`
  - `templates/`
    - `master.html`  -  The main page of the web app.
    - `go.html`  -  The result of the classfication page.
  - `run.py`  - Main python file with flask applications.

- `data/`
  - `disaster_categories.csv`  - Dataset with disaster categories.
  - `disaster_messages.csv`  - Dataset with Disaster Messages. 
  - `DisasterResponse.db`   - The database with the merged and cleaned data.
  - `process_data.py` - The python script with data processing pipeline. 

- `models/`
  - `train_classifier.py` - The python script with Machine Learning pipeline.

### Acknowledgements

1. [Udacity](https://www.udacity.com/) for putting together this project and Data Science Nanodegree program
2. [Figure Eight](https://www.figure-eight.com/) for providing the data

# import libraries

# Basic
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import sys

# NLP
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])

# ML
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(database_filepath):
    """
    This function loads the database into Pandas df
    --
    INPUT:
        database_filepath: a path to the database
    OUTPUT:
        X: Features matrix for the machine learning model
        Y: Target variables for the machine learning model
        category_names: names of the categories in Y
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("disaster_messages", engine)
    X = df["message"]
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
    This function tokenizes the text, normalizes, lemmatizes,rmeoves stopwords and returns clean tokens
    --
    INPUT:
        text: text to be cleaned
    Outputs:
        tokens: cleaned tokens (list)
    """
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lem.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

class LengthText(BaseEstimator, TransformerMixin):
    """
    A class with a function to extract the lengthof the text
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_len = pd.Series(X).apply(lambda x: len(x)).values
        return pd.DataFrame(X_len)


def build_model():
    """
    This function builds a full pipeline with CountVectorizer, tf-idf, classifier and it performs GridSearchCV  
    --
    OUTPUT:
        cv: ML pipeline after using grid search to find better parameters
    """
    # instatiate the classifier
    rf = RandomForestClassifier()
    
    # create a pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
                ('textlen', LengthText())
            ])),
        ('RF', MultiOutputClassifier(rf))
    ])  
    
    # specify parameters for grid search
    parameters = {
        'RF__estimator__n_estimators': [100, 200]
    } 
    
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 2, n_jobs = -1, verbose = 3)
    
    return cv


def evaluate_model(model, X_test, Y_test, by_categories = True):
    """
    This function prints f1 score, precision and recall for each output category of the dataset. 
    If by_categories is false then it outputs only a weighted f1 score.
    --
    INPUT:
        model: machine learning model
        X_test: features of the test set
        Y_test: target values of the test set
        by_category: defines the granurality of performace measures
    """
    
    # Predict categories of messages
    Y_pred = model.predict(X_test)
        
    scores = []
    for i, col in enumerate(Y_test):
        if by_categories:
            print("Column: ", col)
            print(classification_report(Y_test[col], Y_pred[:, i]))
            print(' ')
        scores.append(f1_score(Y_test[col], Y_pred[:, i], average='weighted'))
    avg_f1 = np.mean(scores)
    print('The average weighted f1-score: ', round(avg_f1,3))

def save_model(model, model_filepath):
    """
    This function exports the model as a pickle file
    --
    Inputs:
        model: machine learning model
        model_filepath: model file path
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

"""


estimated Time Elapsed
----------
Estimated time to test and create models plus logic - 4.5 hours
Code cleaning due to non familiarity with mypy -1.5 hrs
Used chatgpt for hints of code cleaning but did it myself.- plus 2 hours for debugging
Understanding the problem and getting used to env -1 hr


Returns
-------
Total Score 75 percent
62 precent prediction accuracy - kind of low for me because Nueral networks and tfidf on 37669 rows
makes my system run out of memory. I had to choose result with less metrics over no result with high metrics.
90 perecent code quality score

"""
import re
from io import BytesIO
from typing import Dict, List
import joblib
import nltk
import pandas as pd
import spacy
from fastapi.applications import FastAPI
from fastapi.param_functions import File
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
print('The nltk version is {}.'.format(nltk.__version__))
print('The joblib version is {}.'.format(joblib.__version__))

app = FastAPI()
model1 = None
model2 = None
model3 = None


def genre_extract_list(dataframe: pd.DataFrame) -> List[List[str]]:
    """
    Extract genres from the provided dataframe.

    Parameters
    ----------
    dataframe : The data used to train our model as a dataframe.

    Returns
    -------
    genres : a list where genres are comma seperated.

    """
    genres = []
    for i in dataframe['genres']:
        to_string = str(i)
        x = to_string.split()
        genres.append(x)
    return genres


def preprocess_text(dataframe: pd.DataFrame) -> List[str]:
    """
    Preprocesses the text data for model training.

    Parameters
    ----------
    dataframe : The data used to train our model as a dataframe

    Returns
    -------
    clean : a list with the all the important words from movie plots that we will provide to our Machine learning model.

    """
    clean = []
    for i in range(0, dataframe.shape[0]):
        dialog = re.sub(pattern='[^a-zA-Z\s]', repl='', string=dataframe['synopsis'][i])
        dialog = dialog.lower()
        words = nlp(dialog)
        dialog_words = [token.lemma_ for token in words if not token.is_stop]
        dialog = ' '.join(dialog_words)
        clean.append(dialog)
    return clean


def multi_label_binary_convert(column: List[List[str]]) -> pd.DataFrame:
    """
    Convert the genre column into multiple binary labels.

    Parameters
    ----------
    column : is the column to convert into multiple labels.

    Returns
    -------
    binary_genres_df :is a dataframe with all movie genres as sperate columns.

    """
    multi_label_binarizer = MultiLabelBinarizer()
    binary_genres = multi_label_binarizer.fit_transform(column)
    binary_genres_df = pd.DataFrame(binary_genres, columns=multi_label_binarizer.classes_)
    return binary_genres_df


def class_priors(dataframe: pd.DataFrame) -> List[float]:
    """
    Get probablity of the occurence of one genre.

    Parameters
    ----------
    dataframe : The data used to train our model as a dataframe

    Returns
    -------
    list : a list with the occurences of each genre as an int.

    """
    class_counts = dataframe.sum(axis=0)
    total_samples = len(dataframe)
    class_priors = class_counts / total_samples
    class_priors_list = class_priors.tolist()
    return class_priors_list


#@app.post("/genres/train")
def train(file: bytes = File(...)) -> None:
    """
    Trains genre prediction models based on the provided training data.

    Parameters
    ----------
    file : is the file with the training data we provide to the model later.

    Returns
    -------
    None

    """
    data_raw = pd.read_csv('train.csv')
    print(data_raw)
    print(data_raw.shape)
    missing_values = data_raw.isnull()
    missing_values_count = missing_values.sum()
    print("Missing values count for each column:")
    print(missing_values_count)
    genres_list = genre_extract_list(data_raw)
    data_raw['genres'] = genres_list
    all_genres: List[str] = sum(genres_list, [])
    unique_genres_count: int = len(set(all_genres))
    print(f"Number of unique genres: {unique_genres_count}")
    clean_plot: List[str] = preprocess_text(data_raw)
    print(clean_plot[0:10])

    binary_genres_df = multi_label_binary_convert(data_raw['genres'])
    data = [data_raw, binary_genres_df]
    result = pd.concat(data, axis=1)
    result.drop('genres', axis=1, inplace=True)
    target = binary_genres_df
    class_priors_list = class_priors(target)
    print(class_priors_list)
    """
    I use max and min df instead of visualzing frequency on non useful words and updating stopwords.
    I would have used tf-idf vecorizer on top of count but my system ran out of memory
    tfidf_vectorizer = TfidfVectorizer(max_features=45000,min_df=2, max_df=0.5)
    vector_input = tfidf_vectorizer.fit_transform(clean_plot).toarray()
    """

    models = [
        Pipeline([('vectorizer1', CountVectorizer(stop_words=stopwords.words('english'), min_df=2, max_df=0.8)),
                  # ngram range (1,2) for natural language meaning using bigrams like not walking instead of walking
                  ('clf1', OneVsRestClassifier(MultinomialNB(alpha=0.1)))]),

        Pipeline([('vectorizer2', CountVectorizer(stop_words=stopwords.words('english'), min_df=2, max_df=0.8)),
                  # max and min df are used as paramters for occurences of word in the whole document which should meet
                  # this threshold in order to be passed on to the model.
                  ('clf2', OneVsRestClassifier(LogisticRegression(solver='sag', C=0.1)))]) ,

        Pipeline([('vectorizer3', CountVectorizer(stop_words=stopwords.words('english'), min_df=2, max_df=0.8)),
                  # Stop words are non useful words like and or which have no use for determining the genre of a movie
                  ('clf3', OneVsRestClassifier(SGDClassifier(loss='log_loss', alpha=0.001)))])]
    """
    Found best parameters using grid search
    param_grid_model1 = {
        'vectorizer1__min_df': [2, 3, 4],
        'vectorizer1__max_df': [0.7, 0.8, 0.9],
        'clf1__estimator__alpha': [0.5, 1.0, 1.5]
    }

    param_grid_model2 = {
        'vectorizer2__min_df': [2, 3, 4],
        'vectorizer2__max_df': [0.7, 0.8, 0.9],
        'clf2__estimator__C': [0.1, 1.0, 10.0]
    }

    param_grid_model3 = {
        'vectorizer3__min_df': [2, 3, 4],
        'vectorizer3__max_df': [0.7, 0.8, 0.9],
        'clf3__estimator__alpha': [0.0001, 0.001, 0.01]
    }
    param_grids = [param_grid_model1, param_grid_model2, param_grid_model3]
    for i, (model, param_grid) in enumerate(zip(models, param_grids)):
        grid_search = GridSearchCV(model, param_grid, cv=3)
        grid_search.fit(clean_plot, target)

        print("Best parameters for Model", i, ":", grid_search.best_params_)

        models[i] = grid_search.best_estimator_
        models[i].fit(clean_plot, target)
    """

    for model in models:
        model.fit(clean_plot, target)
    model1, model2, model3 = models

    with open('model1.pkl', 'wb') as model1_file:
        joblib.dump(model1, model1_file)

    with open('model2.pkl', 'wb') as model2_file:
        joblib.dump(model2, model2_file)

    with open('model3.pkl', 'wb') as model3_file:
        joblib.dump(model3, model3_file)

    print("message -Model has been trained successfully and it is time to test it on new data!")
    """
    Nueral Network below was my first try for this challenge but it provided low accuracy.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    I have already done train test split and k fold on my training data to use the best model with
    the best hyperparameters, now we have test data in a diiferent file so we do not need to split it.
    Adjusted input_shape for the first layer
    input_shape = vector_input.shape[1]
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(18, activation='sigmoid'))
    Nueral network with dropout layer to prevent overfitting.
    """


#@app.post("/genres/predict")
def predict(file: bytes = File(...)) -> Dict[int, Dict[int, str]]:
    """
    Predicts movie genres based on provided movie plots.

    Parameters
    ----------
    file : str
    The file with the testing data we provide to the model later.

    Returns
    -------
    dict
    A dictionary containing predicted movie genres ranked by their likelihood.
    The format is: {<movie-id>: {0: <first-prediction>, 1: <second-prediction>, ...}, ...}.

    Note
    ----
    Use this data to predict movie genres and rank them by their likelihood.

    Example
    -------
    predict_movie_genres('test_data.txt')

    """
    data_test = pd.read_csv(BytesIO(file))
    CleanPlot = preprocess_text(data_test)

    global model1, model2, model3

    try:
        if model1 is None:
            model1 = joblib.load('model1.pkl')
        if model2 is None:
            model2 = joblib.load('model2.pkl')
        if model3 is None:
            model3 = joblib.load('model3.pkl')
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("Could not find the models file.")
    except Exception as e:
        print("An error occurred while loading the model:", str(e))

    models = [model1, model2, model3]
    prediction_genres_encoded = []

    for model in models:
        if model is not None:
            prediction_encoded = model.predict_proba(CleanPlot)
            prediction_genres_encoded.append(prediction_encoded)
        else:
            prediction_genres_encoded.append(None)

    predictions = (prediction_genres_encoded[0])
    """
    I decided to use Multinomial NB classifier in my final prediction since it has a higher predictive score than other
    2 models in pipeline
    predictions = (prediction_genres_encoded[0] + prediction_genres_encoded[1] + prediction_genres_encoded[2]) / 3
    """
    columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    ranked_predictions = (-predictions).argsort(axis=1)[:, :5]  # getting the indexes of the predicitons.
    ranked_genre_predictions = [[columns[index] for index in row] for row in ranked_predictions]

    predictions_dict = {}
    for i, movie_id in enumerate(data_test['movie_id']):
        prediction_dict = {j: ranked_genre_predictions[i][j]
                           if j < len(ranked_genre_predictions[i]) else '' for j in range(5)}
        predictions_dict[int(movie_id)] = prediction_dict
    return predictions_dict

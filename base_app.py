"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
additional  = ['retweet']
stop = set().union(stopwords.words('english'),additional)

#IMPORTS
import pandas as pd
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re # for regular expressions
import pandas as pd 
pd.set_option("display.max_colwidth", 200) 
import string
special = string.punctuation 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy import stats 
from sklearn import metrics 
from sklearn.metrics import mean_squared_error,mean_absolute_error, make_scorer,classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm

import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt



# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def TweetCleaner(tweet):
    
    #This function uses regular expressions to remove url's, mentions, hashtags, 
    #punctuation, numbers and any extra white space from tweets after converting everything to lowercase letters.
    
    # Convert everything to lowercase
    tweet = tweet.lower() 
    # Remove mentions   
    tweet = re.sub('@[\w]*','',tweet)  
    # Remove url's
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)    
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)  
    # Remove punctuation
    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", ' ', tweet)
    # Remove that funny diamond
    tweet = re.sub(r"U+FFFD ", ' ', tweet)
    # Remove extra whitespace
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # Remove space in front of tweet
    tweet = tweet.lstrip(' ')
    # Remove emojis
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    #Remove stop words
    remove_stopwords = [w for w in tweet.split() if w not in stop]
    tweet = ' '.join(remove_stopwords)
    
    return tweet

# Clean the tweets in the message column
raw['clean_message'] = raw['message'].apply(TweetCleaner)



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Cleaned Twitter data and label")
		if st.checkbox('Show cleaned data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'clean_message']]) # will write the df to the page            

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		option = ["Stochastic Gradient Descent", "Logistic Regression","Support Vector Machines"]
		select = st.sidebar.selectbox("Choose Model", option)
        
		if select =="Logistic Regression":        
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				# Creating sidebar with selection box -
				# you can create multiple pages this way
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == -1:
					st.success("Text Categorized as:  Anti Climate Change")
				if prediction == 0:
					st.success("Text Categorized as:  Neutral, Neither Believe Nor Disputes Climate Change")
				if prediction == 1:
					st.success("Text Categorized as:  Pro Climate Change, Believe In Climate Change")
				if prediction == 2:
					st.success("Text Categorized as:  Factual News On Climate Change")      

		if select =="Stochastic Gradient Descent":
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				# Creating sidebar with selection box -
				# you can create multiple pages this way
				predictor = joblib.load(open(os.path.join("resources/SGD_Classifier_AM5_DSFT21.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == -1:
					st.success("Text Categorized as:  Anti Climate Change")
				if prediction == 0:
					st.success("Text Categorized as:  Neutral, Neither Believe Nor Disputes Climate Change")
				if prediction == 1:
					st.success("Text Categorized as:  Pro Climate Change, Believe In Climate Change")
				if prediction == 2:
					st.success("Text Categorized as:  Factual News On Climate Change")
            
		if select == "Support Vector Machines":
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				# Creating sidebar with selection box -
				# you can create multiple pages this way
				predictor = joblib.load(open(os.path.join("resources/Support_Vector_Machine_AM5_DSFT21.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == -1:
					st.success("Text Categorized as:  Anti Climate Change")
				if prediction == 0:
					st.success("Text Categorized as:  Neutral, Neither Believe Nor Disputes Climate Change")
				if prediction == 1:
					st.success("Text Categorized as:  Pro Climate Change, Believe In Climate Change")
				if prediction == 2:
					st.success("Text Categorized as:  Factual News On Climate Change")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

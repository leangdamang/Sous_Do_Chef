import flask
import pandas as pd
import numpy as np


from numpy import random

from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

#Load in all the data
tfmodel = joblib.load("/Users/leangchaing/ds/metis/metisgh/fletcher/webapp/pickles/tfmodel")
df = pd.read_pickle("/Users/leangchaing/ds/metis/metisgh/fletcher/webapp/pickles/df_posts")
corpus = joblib.load("/Users/leangchaing/ds/metis/metisgh/fletcher/webapp/pickles/tfvectors")
nmfmodel = joblib.load("/Users/leangchaing/ds/metis/metisgh/fletcher/webapp/pickles/nmffit")


null_responses = ["Sorry I have no clue, I'm dumb", "Ask again later", "Malfunctioning. Malfunctioning"]
stop = stopwords.words('english')
stop += ['.', ',', '(', ')', "'", '"', 'com', 'http', 'www', 'https', 'youtube', 'watch', 'video',
         'imgur', 'watch', 'imgur', 'gallery', 'jpg', 'like', 'make', 'cooking', 'would', 'cook', 'recipe', 'recipes',
         'good', 'use', 'one', 'get', 'know', 'want', 'really', 'something', 'help', 'thanks', 'need', 'anyone',
         'looking', 'making', 'way', 'much', 'ideas', 'made', 'add', 'also', 'could', 'using', 'food', 'go',
         'things', 'got', 'going', 'think', 'people', 'lot', 'work', 'day', 'well', 'try', 'find',
         'put', 'even', 'edit', 'anything']

# Initialize the app
app = flask.Flask(__name__)

@app.route("/")
def viz_page():
    with open('Sous_do_Chef.html', 'r') as viz_file:
        return viz_file.read()

@app.route("/culinary", methods=["POST", "GET"])

def answer():
    print("Let's get to cooking")
    query = flask.request.json
    queryentry = query['question']
    df_answer = get_predict_df(queryentry[0])

    if df_answer.iloc[0]['similarity'] <= .1:
        answer = null_responses[random.randint(len(null_responses)-1)]
        return flask.jsonify({'answer': answer})
    else:
        a = ' '.join(df_answer.iloc[0][['top_comment', 'second_comment']])
        summary = summarize(a, 3)
        answer = ''
        for s in summary:
            answer+= str(s) + '\n'
        return flask.jsonify({'answer': answer})

def get_predict_df(text):
    textl = []
    textl.append(text)
    df_cat = df[df['topic'] == get_category(textl, tfmodel, nmfmodel)].reset_index()
    cattf = TfidfVectorizer(stop_words=stop, ngram_range=(1,2), min_df = 2, max_df = .95, strip_accents = 'ascii')
    cattf.fit(df_cat['question'])
    query_vector = cattf.transform(df_cat['question'])
    text_vec = cattf.transform(textl)
    df_cat['similarity'] = pd.DataFrame(cosine_similarity(query_vector, text_vec))
    return df_cat.sort_values('similarity', ascending = False).reset_index()

def summarize(answer, n_sentences):
    summarizer = TextRankSummarizer()
    parse = PlaintextParser.from_string(answer, Tokenizer("english"))
    summary = summarizer(parse.document, n_sentences)
    return summary

def get_category(query, tf, model):
    query_vector = tf.transform(query) #transforms it to the tfidf
    return get_top_topic(model.transform(query_vector))

def get_top_topic(model):
    return np.argmax(model)

app.run(host='0.0.0.0', debug=True)

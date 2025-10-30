import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv("tripadvisor_hotel_reviews.csv")

print(data.info())
print(data.head())
print(data['Review'][0])

data['review_lowercase'] = data['Review'].str.lower()

en_stopwords = stopwords.words('english')
if "not" in en_stopwords:
    en_stopwords.remove("not")

data['review_no_stopwords'] = data['review_lowercase'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in en_stopwords])
)

data['review_no_stopwords_no_punct'] = data.apply(
    lambda x: re.sub(r"[*]", "star", x['review_no_stopwords']), axis=1
)
data['review_no_stopwords_no_punct'] = data.apply(
    lambda x: re.sub(r"([^\w\s])", "", x['review_no_stopwords_no_punct']), axis=1
)

data['tokenized'] = data.apply(
    lambda x: word_tokenize(x['review_no_stopwords_no_punct']), axis=1
)

ps = PorterStemmer()
data['stemmed'] = data['tokenized'].apply(lambda tokens: [ps.stem(token) for token in tokens])

lemmatizer = WordNetLemmatizer()
data['lemmatized'] = data['tokenized'].apply(
    lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
)

tokens_clean = sum(data['lemmatized'], [])

unigrams = pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()
bigrams = pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()
ngrams_4 = pd.Series(nltk.ngrams(tokens_clean, 4)).value_counts()

print(data[['Review', 'review_lowercase', 'review_no_stopwords_no_punct', 'tokenized']].head())
print(unigrams.head())
print(bigrams.head())
print(ngrams_4.head())

print("Text preprocessing completed successfully!")

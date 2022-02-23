#%%
#INICIO DE SECION A LA APLICACION 
import tweepy
import json

api_key="75Za10xzfxcQbenx6IAjnDwyn"
api_key_secret="yg7EeuqnMEC6ovNFjL8HQQSCOl1yMoMtndqrfwZuBa1MDBXo0s"
access_token="1495874107157716995-AwW9VYibRkBDDrz9XxIPmNIEq2ImN3"
access_token_secret="eb2atoH9o3AxmewItpwr0Iqb22CIzWr6BJkMJM4tKtrW4" 

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api=tweepy.API(auth,wait_on_rate_limit=True)
data=api.verify_credentials()
print(json.dumps(data._json,indent=2))


# %%
#Obtencion de tweets sobre Franciamarquez
import csv

file = open('francia.csv', 'w',encoding='utf-8')
csvWriter = csv.writer(file)
end_date="2022-02-22"
for tweet in tweepy.Cursor(api.search_tweets, q="@FranciaMarquezM" ,
                                       lang="es",
                                       since="2018-01-01",until= end_date ).items():
                csvWriter.writerow([tweet.created_at, tweet.text])

file.close()
#############################################################################3
###########################################################################33

# %%
# cargar tweets  del cvs 
import pandas as pd
data = pd.read_csv("francia.csv", header = None, encoding='utf-8', names = ['Time', 'Tweets'])
data

# %%
# Borrando duplicados de los tweets
dfaux= data.drop_duplicates(subset=['Tweets'])

dfaux

#%%
# Limpieza de tweets
from unicodedata import normalize
import re
import string

dfaux["Tweets_limpios"]=data["Tweets"]
# #removing hashtags related to globalwarming
def rem_hashtags(text):
    processed_text = re.sub(r"#@FranciaMarquezM", "", text)
    processed_text = " ".join(processed_text.split())
    return processed_text
dfaux["Tweets_limpios"] = dfaux["Tweets_limpios"].apply(lambda x:rem_hashtags(x))

#removing tagged users from the tweets
def remove_users(text):
    processed_text = re.sub(r'@\w+ ?',"",text)
    processed_text = " ".join(processed_text.split())
    return processed_text
dfaux["Tweets_limpios"] = dfaux["Tweets_limpios"].apply(lambda x:remove_users(x))

#removing hyperlinks mentioned in the tweets
def remove_links(text):
    processed_text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
    processed_text = " ".join(processed_text.split())
    return processed_text
dfaux["Tweets_limpios"] = dfaux["Tweets_limpios"].apply(lambda x:remove_links(x))

#making all tweets lowercase
def lowercase_word(text):
    text  = "".join([char.lower() for char in text])
    text = re.sub(r"rt", "", text)
    punct = string.punctuation
    text= re.sub(r"%punct","", text)
    
    return text
dfaux["Tweets_limpios"] = dfaux["Tweets_limpios"].apply(lambda x: lowercase_word(x))

def normalizarP(text):
    palabra = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
             normalize("NFD",text), 0, re.I)
    palabra= normalize('NFC',palabra)
    return palabra

dfaux["Tweets_limpios"] = dfaux["Tweets_limpios"].apply(lambda x: normalizarP(x))


dfaux

# %%
# Uso de NLTK
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
texto="Esto es un texto de prueba para ver el funcionamiento, de la libreria NLTK ademas de como se tokeniza las palabras "
stop_words =set(stopwords.words('spanish'))

def Tokenizar(texto):
    word_tokens = word_tokenize(texto)
    word_tokens =list(filter(lambda token: token not in string.punctuation ,word_tokens))
    filtro=[ palabra for  palabra in word_tokens if palabra not in stop_words]
    tokens=",".join(filtro)
    return tokens

dfaux["Tokens"]=dfaux["Tweets_limpios"].apply(lambda x: Tokenizar(x))

dfaux
# %%
# Lematizacion
import spacy
nlp = spacy.load('es_core_news_sm')

def lematizar(text):
    doc= nlp(text)
    tokens =[]
    for token in doc:
        if token.lemma_ !="-PRON-":
            temp =token.lemma_.strip()
        else:
            temp =token
        tokens.append(temp) 
    clean_tokens=[]
    for token in tokens:
        if token not in stop_words and token not in string.punctuation:
            clean_tokens.append(token)
    lematizado=",".join(clean_tokens)
    return lematizado  

dfaux["Tokens lematizados"]=dfaux["Tokens"].apply(lambda x:lematizar(x))              

# %%
dfaux["Tokens lematizados"]
# %%Analisis de sentimiento
from  textblob import TextBlob

def  get_tweet_sentiment(text):
    valor=TextBlob(text).sentiment.polarity
    if valor > 0: 
        return 'positive'
    elif valor == 0: 
        return 'neutral'
    else: 
        return 'negative'

dfaux["sentimiento"]=dfaux["Tweets_limpios"].apply(get_tweet_sentiment)

dfaux

# %%
print(len(dfaux[dfaux["sentimiento"]=="positive"]))
print(len(dfaux[dfaux["sentimiento"]=="neutral"]))
print(len(dfaux[dfaux["sentimiento"]=="negative"]))
# %%
dfaux["sentimiento"].hist()

# %%

#importing of packages
from nltk import tokenize
from nltk.stem.api import StemmerI
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#read a csv file
df=pd.read_csv("spam.csv",encoding= 'unicode_escape')

#creating a column for spam
df["spam"]=df["type"].map({"spam":1,"ham":0})

stemmer=SnowballStemmer("english",ignore_stopwords=False)
stop_word=stopwords.words("english")
wnl=WordNetLemmatizer()
tfidf=TfidfVectorizer()

def preprocess(text):
    tokens=[w for w in word_tokenize(text.lower()) if w.isalpha()]
    text=[t for t in tokens if t not in stop_word]
    return ' '.join(text)

def stemming(text):
    text=stemmer.stem(text)
    return ''.join(text)

def lemmitazer(text):
    text=wnl.lemmatize(text)
    return ''.join(text)

df["text"]=df["text"].apply(preprocess)
df["text"]=df["text"].apply(stemming)
df["text"]=df["text"].apply(lemmitazer)
print(df)

X=tfidf.fit_transform(df['text'])
y=df['spam']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10,shuffle=False)

svc=LinearSVC()
svc.fit(x_train,y_train)
pred=svc.predict(x_test)
linear_accuracy=metrics.accuracy_score(y_test,pred)
print('Accuracy from Linear SVC:',linear_accuracy*100)

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(x_train, y_train)
pred = Spam_model.predict(x_test)
log_accuracy=metrics.accuracy_score(y_test,pred)
print('Accuracy from Logistic Regression:',log_accuracy*100)


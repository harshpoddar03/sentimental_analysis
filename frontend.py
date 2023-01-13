#importing numpy and pandas to store, manipulate and process data
import pandas as pd
import numpy as np
#importing regular expressions librarires for pre- processing 
import re, string, unicodedata
#importing these libraries for pre-processing purposes
#use pip install contractions to install them
import contractions
import inflect
#importing bs4 to identify and remove web-tags
from bs4 import BeautifulSoup
#nltk is used for natural language processing
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
#importing pre-processing libraries from keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
#importing keras to access machine learning model
from keras.models import Sequential
#importing keras for defining neutral netwrok
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
#importing sklearn 
from sklearn.model_selection import train_test_split
import keras
from scipy.stats import gamma 
from nltk.corpus import stopwords
from tkinter import *
from PIL import ImageTk,Image

#this helps in the pre processing of the data and seperate the words which do not cause any sentiment
from nlppreprocess import NLP

stop_words= stopwords.words('english')

nlp = NLP()
from tensorflow import keras
model = keras.models.load_model('D:\sentiment-analysis-GUI\my_model2')


def result(a):
  a=pre_pro(a)
  X=np.array([a])

  X = tokenizer.texts_to_sequences(X)

  vocab_size = len(tokenizer.word_index) + 1
  maxlen = 25
  X = pad_sequences(X, padding='post', maxlen=maxlen)

  x=model.predict(X)[0]
  print (x)

  if x[1]>x[0]:
      print ('positive')
  else:
      print ("negative")
  return x    


tokenizer = Tokenizer()
import pickle
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)


z=Tk()
z.title("Sentimental analysis ")
x=0
def add():
    global x
    x=x+1
    if x==1:
        frame.pack_forget()
        frame2.pack(side="top",expand=True,fill="both")
        
def subt():
    global x
    x=x-1
    if x==0: 
        frame2.pack_forget()
        frame.pack(side="top",expand=True,fill="both")
        


#INTROFRAME
frame=LabelFrame(z)
frame.pack(side="top",expand=True,fill="both")
frame.configure(bg="grey26")
imga=ImageTk.PhotoImage(Image.open("C:\\Users\\podda\\OneDrive\\Desktop\\abcd.jpg"))
imgal=Label(frame,image=imga)
imgal.pack()
title= Label(frame, text="SENTIMENT \n ANALYSIS \n WITH DEEP \n LEARING",bg="white",fg="deep sky blue",font=("Eras Bold ITC",50,"bold"))
title.place(relx=0.5,rely=0.5,anchor='center')
dest=Button(frame,text="EXIT",font=("impact"),fg="deep sky blue",bg="white",width=10,command=z.destroy)
dest.place(relx=0.1,rely=0.9,anchor="s")
conti=Button(frame,text="CONTINUE",fg="deep sky blue",bg="white",font="impact",width=10,command=add)
conti.place(rely=0.9,relx=0.9,anchor="s")



####FRAME 1
frame2=LabelFrame(z)
frame2.configure(bg="lavender")
imga2=ImageTk.PhotoImage(Image.open("C:\\Users\\podda\\OneDrive\\Desktop\\champak.jpg"))
imgal2=Label(frame2,image=imga2)
imgal2.pack()
frameser=LabelFrame(frame2)
search=Entry(frameser,width=20,font=("century gothic",25),fg="grey")

def backpb():
    framep.pack_forget()
    frame2.pack(side="top",expand=True,fill="both")

def backnb():
    framen.pack_forget()
    frame2.pack(side="top",expand=True,fill="both")



def btsent():
    result=search.get()
    print(result)
    if result=="positive":
        frame2.pack_forget()
        framep.pack()
    elif result=="negitive":
        frame2.pack_forget()
        framen.pack()



#FRAME2
framep=LabelFrame(z)
imgap=ImageTk.PhotoImage(Image.open("123.jpg.jpeg"))
imgalp=Label(framep,image=imgap)
imgalp.pack()
title= Label(framep, text="POSITIVE ",bg="#F1948A",fg="deep sky blue",font=("Eras Bold ITC",50,"bold"))
title.place(relx=0.5,rely=0.1,anchor='center')
backp=Button(framep ,text="BACK",fg="deep sky blue",bg="WHITE",font="impact",command=backpb)
backp.place(rely=0.9,relx=0.9)

#FRAME3
framen=LabelFrame(z)
imgan=ImageTk.PhotoImage(Image.open("234.jpg.jpeg"))
imgaln=Label(framen,image=imgan)
imgaln.pack()
title= Label(framen, text="NEGATIVE ",bg="RED",fg="deep sky blue",font=("Eras Bold ITC",50,"bold"))
title.place(relx=0.5,rely=0.1,anchor='center')
backn=Button(framen ,text="BACK",fg="DEEP SKY blue",bg="WHITE",font="impact",command=backnb)
backn.place(rely=0.9,relx=0.9)



def exec():
  
  a=search.get()
  x=result(a)
  if x[1]>x[0]:
    frame2.pack_forget()
    framep.pack()

      
  else:
    frame2.pack_forget()
    framen.pack()
      



searchb=Button(frameser,text="PREDICT",fg="DEEP SKY blue",bg="WHITE",font="impact",command=exec)
frameser.place(rely=0.4,relx=0.3)
search.grid(row=1,column=2)
searchb.grid(row=1,column=3)
back=Button(frame2,text="BACK",fg="DEEP SKY blue",bg="WHITE",font="impact",command=subt)
back.place(rely=0.9,relx=0.9)



def pre_pro(text):
  
  
  
#HTML tags were found in the text so Bs4 was used to strip them to get better sentiments as HTML tags are <br> doesnt help in sentiment analysis
  def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


#removing the square brackets as it increase inconsistencies in pre-processing
  def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)



  def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


#we have to remove contraction to get clear idea of the text. like 'you'll' will get converted into 'you will'
  def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)    


  #removing indentified general pattern names starting with @ and #tags. when we scrap a data from the web it may contain symbols like '@' and '%' which do not tell us about the setiments of the sentences .Thus they are removed. 
  text=" ".join (word for word in text.split() if word[0] not in ['@', '#'])
  
  #denoise
  
  text=denoise_text(text)
  
  #removing stop words. stop wods are words like 'and' , 'if' etc which do not have any role in setimental analysis hence they are removed.
#   print(text)
#   text=" ".join (word for word in text.split() if word not in stop_words
  text=nlp.process(text)

  #removing line skip. replaces the unneeded elements by blankspace
  text=text.replace('[^\w\s]','')
  
  # Remove punctuations and numbers. punctuation like !, ? and numbers are not used in sentimental analysis 
  text = re.sub('[^a-zA-Z]', ' ', text)
  
  #lowercase. Changing into lower case for uniform datastructure
  text=" ".join (word.lower() for word in text.split())
 
 #contraction.
  text=replace_contractions(text)

  # Single character removal
  text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

  # Removing multiple spaces
  text = re.sub(r'\s+', ' ', text)

  return text

z.mainloop()
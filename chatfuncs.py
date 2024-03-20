
import transformers 
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


saved_model = pipeline('text-classification',
                       model = 'training_dir/checkpoint-50')

def replace(txt):
    txt=txt.replace('midterm','mid term')
    txt=txt.replace('retest','re test')
    txt=txt.replace('make up','makeup')
    txt=txt.replace('vii','seventh')
    txt=txt.replace('iii','third')
    txt=txt.replace('ii','second')
    
    txt=txt.replace(' iv ',' fourth ')
    txt=txt.replace(' v ',' fifth ')
    txt=txt.replace(' vi ',' sixth ')
    
    txt=txt.replace(' i ',' first ')
    
    
    return txt


ls=WordNetLemmatizer()
translator=str.maketrans('','',string.punctuation)
def preprocess_text(txt):
    txt=txt.lower()
    txt=txt.translate(translator)
    txt=txt.replace('\n',' ')
    
    txt=replace(txt)
    
    stop_words = set(stopwords.words('english'))
    tokens=[x for x in word_tokenize(txt) if x not in stop_words]
    
    t=""
    for i in tokens:
        t+=ls.lemmatize(i)+" "
    t=t.strip()
    return t



from nltk.corpus import words
eng_words = words.words()

def check_vocab(txt):
    tokens= word_tokenize(txt)
    for w in tokens:
        if w not in eng_words:
            return (w, False)
    return (w,True)

def direct_search(txt):
    for i in df.select_dtypes(include='object').columns:
            for j in df[i]:
                if type(j)!=float and len(j)>0:
                    j=preprocess_text(j)
                    if txt in j:
                        return f'{i}:{j}'
                    
df=pd.read_excel('AttachmentMIT-Academic_Calendar_2023-24.-1 (2).xlsx')
df.drop([0,1],inplace=True)
df.columns=df.iloc[0]
df.drop(2,inplace=True)
l=[]
for i in df.columns:
    for j in df[i]:
        if j!=None and type(j)!=float:
            l.append(j)                
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()

def calc_sim(t1,t2):
    emb1=tfidf.fit_transform([t1])
    emb2=tfidf.transform([t2])
    sim= cosine_similarity(emb1,emb2)
    return sim


nlp=pipeline('conversational',model='base_chatbot')



def chat(input_text):

    
    if input_text:
        if input_text.lower() == 'bye':
            return 'Goodbye'
            
        input_text = input_text.lower()
        # input_text = spellcheck(input_text)
        input_text = preprocess_text(input_text)            
        label = saved_model(input_text)[0]['label']
        f = 0

        if label == 'LABEL_1':
            w, b = check_vocab(input_text)
            if not b:
                return direct_search(w)
            else:
                    # Assume 'df' is your DataFrame
                for i in df.select_dtypes(include='object').columns:
                    for j in df[i]:
                        if type(j) != float and len(j) > 0:
                            j = preprocess_text(j)
                            
                            sim = calc_sim(input_text, j)
                            print(" assistant:"+str(j)+','+str(sim))
                            if sim > 0.6:
                                
                                return f'{i}:{j}'
               
            return 'Could not find any related event'
          
        else:
            return nlp(transformers.Conversation(input_text), pad_token_id=50256)[1]['content'] 
    


        


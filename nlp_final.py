# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 00:08:20 2022

@author: w2000
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:05:15 2022

@author: w2000
"""

'''
import os, time
import jieba
import jieba.analyse 
import jieba.posseg as psg
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.probability import FreqDist
import random
from nltk import classify
from nltk import NaiveBayesClassifier
import re
'''

'''
#text 
character=["Dudley", "Hermione", "Harry", "Malfoy"]
stopwords=[]
line_with_name=[]
words=[]

#create a stopwords list
for i in range (1, 4):
    stopwords.append(character[i])
for stopword in open(file="stopwords_Dudley.txt" , mode="r", encoding="utf8"):
    stopwords.append(stopword.strip())


#to get the lines with descriptions of certain character (the lines with that that character's name)
for i in range (0, 1):        
    with open(file="harry_potter.txt", mode="r", encoding="utf8") as file1:
        text=file1.read().replace("\n", " ")
        cutsentence=re.split(r'[\n;.?!]', text) 
        for sentence in cutsentence:
            a=re.search(pattern=character[i], string=sentence)
            if a:
                line_with_name.append(sentence)
    with open(file="%s.txt" %character[i], mode="w", encoding="utf8") as file2:
        for sentence in line_with_name:
            cutword=psg.cut(sentence)
            for word in cutword:
                if word.word not in stopwords:
                    words.append(word.word)
        file2.write(" ".join(words).strip())


#get features of a character with v2w and do further stopwords removal
train_data = word2vec.LineSentence('Dudley.txt')    #PercyLineSentence: path to a corpus file in
model = word2vec.Word2Vec(
    train_data,
    min_count=1,                                    #ignores all words with total frequency lower than this
    vector_size=100,                                #dimensionality of the word vectors
    workers=8,                                      #use these many worker threads to train the model; namely to train these many group of data at the same time
    epochs=5,                                       #number of iterations (epochs) over the corpus
    window=20,                                      #the number of surrounding voc to be checked
    sg=0,                                           #w2v includes 2 operation, CBOW(0) &Skip-gram(1), and CBOW has higher training speed
    seed=519,                                       #random number generator
)
    

model.save('Dudley.model')     
model = word2vec.Word2Vec.load('Dudley.model')
print(model.wv['Dudley'].shape)                     #without shape, all vector data will show up 
for item in model.wv.most_similar('Dudley'):
    print(item)  
'''
'''
#DISC classifier
stopwords=[]
for stopword in open(file="DISC_stopwords.txt", mode="r", encoding="utf8"):
    stopwords.append(stopword.strip()) #.strip('x') to cut off all the char x at the beginning and the end of the str
                                       #.strip() is to remove blank spaces at the beginning/end of the str
                                       #remove .strip() \n would show up


#remove stopwords 
personality=['D','I','S','C']
personalityfile=['d', 'i', 's','c']
personalityok=['Dok', 'Iok', 'Sok','Cok']
totality=49
for i in range(0, 4):    
    for j in range (1, totality+1):
        file_txt1="%c\%c%d.txt" %(personality[i], personalityfile[i], j)
        with open(file=file_txt1, mode="r", encoding="utf8") as file1:
            one_txt=file1.read().replace("\n", " ") 
            cutword = psg.cut(one_txt) 

            file_txt2="%s\%s%d.txt" %(personalityok[i], personalityok[i], j)
            with open(file=file_txt2, mode="w", encoding="utf8") as file2:
                words=[]
                for word in cutword:
                    if word.word not in stopwords:
                        words.append(word.word)
                file2.write(" ".join(words).strip()) #.join: to add new elements in a str
                    
#the keywords of the target might be included in the stopwords 
#stopwords emit the words based on the part of speech                      
#pos, x, means unknown
#pas, eng, indicates english. namely the system is supposed to deal with chinese
#it's the problem of encoding. utf8 is meant for encoding Chinese
'''
#D
c="Dok"
pcr_D = PlaintextCorpusReader(root=c, fileids=".*\.txt") 
D_doc = [(pcr_D.words(fileid),c) for fileid in pcr_D.fileids()]

#I
c="Iok"
pcr_I = PlaintextCorpusReader(root=c, fileids=".*\.txt") 
I_doc = [(pcr_I.words(fileid),c) for fileid in pcr_I.fileids()]

#S
c="Sok"
pcr_S = PlaintextCorpusReader(root=c, fileids=".*\.txt") 
S_doc = [(pcr_S.words(fileid),c) for fileid in pcr_S.fileids()]

#C
c="Cok"
pcr_C = PlaintextCorpusReader(root=c, fileids=".*\.txt") 
C_doc = [(pcr_C.words(fileid),c) for fileid in pcr_C.fileids()]
#[(['CONSCIENTIOUSNESS', 'cautious', 'skeptical', ...], 'Cok')]

c="Hermione"
pcr_her = PlaintextCorpusReader(root="./", fileids="Hermione.txt") 
her_doc = [(pcr_C.words(fileid),c) for fileid in pcr_C.fileids()]
#print(her_doc)


documents = D_doc + I_doc + S_doc + C_doc
random.shuffle(x=documents) 

N_features =100
all_words = FreqDist(pcr_D.words() + pcr_I.words() + pcr_S.words() + pcr_C.words())   
word_features = list(all_words)[:N_features] 
#print(word_features)


def document_features(document_words):
    document_words = set(document_words)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words) 
    return features


N_testing = 20
featuresets = [(document_features(d), c) for (d,c) in documents] 
#print(featuresets[:1])

her_features=[(document_features(d), c) for (d,c) in her_doc] 
#print(her_features)

  
train_set, test_set = featuresets[N_testing:], featuresets[:N_testing]


classifier = NaiveBayesClassifier.train(train_set) 
#A ELE probability distribution must have at least one bin.

#print(classify.accuracy(classifier, test_set))
#print(classifier.show_most_informative_features(5))


print(classifier.classify(her_features)) #her_features from line 163





#words = [word for word, pos in tagged_words if pos not in ['m']] 
#            words=[]
#            for word, pos in tagged_words:
#                if pos not in ['m']:
#                    words.append(word)

#list=[1, 5, 6, 7]
#newlist=[x**2 for x in list]
#newlist: 1, 25, 36, 49




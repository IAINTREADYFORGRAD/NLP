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
'''

stopwords=[]
for stopword in open(file="DISC_stopwords.txt", mode="r", encoding="utf8"):
    stopwords.append(stopword.strip()) #.strip('x') to cut off all the char x at the beginning and the end of the str
                                       #.strip() is to remove blank spaces at the beginning/end of the str
                                       #remove .strip() \n would show up

personality=['D','I','S','C']
personalityfile=['d', 'i', 's','c']
personalityok=['Dok', 'Iok', 'Sok','Cok']
totality=49
for i in range(0, 4):    
    for j in range (1, totality+1):
        file_txt1="%c\%c%d.txt" %(personality[i], personalityfile[i], j)
        with open(file=file_txt1, mode="r", encoding="utf8") as file1:
            one_txt=file1.read().replace("\n", " ") 
            
            file_txt2="%c\collect_%c.txt" %(personality[i], personalityfile[i])
            with open(file=file_txt2, mode="a", encoding="utf8") as file2:
                file2.write(one_txt)
        
    with open(file=file_txt2, mode="r", encoding="utf8") as file2:
        cutword = psg.cut(file2.read()) 
    
        file_txt3="%c\%s.txt" %(personality[i], personalityok[i])
        with open(file=file_txt3, mode="w", encoding="utf8") as file3:
            words=[]
            for word in cutword:
                if word.word not in stopwords:
                    words.append(word.word)
            file3.write(" ".join(words).strip()) #.join: to add new elements in a str
                    
#the keywords of the target might be included in the stopwords 
#stopwords emit the words based on the part of speech                      
#pos, x, means unknown
#pas, eng, indicates english. namely the system is supposed to deal with chinese
#it's the problem of encoding. utf8 is meant for encoding Chinese

#D
c="D"
pcr_D = PlaintextCorpusReader(root=c, fileids="Dok.txt") 
D_doc = [(pcr_D.words(fileid),c) for fileid in pcr_D.fileids()]

#I
c="I"
pcr_I = PlaintextCorpusReader(root=c, fileids="Iok.txt") 
I_doc = [(pcr_I.words(fileid),c) for fileid in pcr_I.fileids()]

#S
c="S"
pcr_S = PlaintextCorpusReader(root=c, fileids="Sok.txt") 
S_doc = [(pcr_S.words(fileid),c) for fileid in pcr_S.fileids()]

#C
c="C"
pcr_C = PlaintextCorpusReader(root=c, fileids="Cok.txt") 
C_doc = [(pcr_C.words(fileid),c) for fileid in pcr_C.fileids()]
#[(['CONSCIENTIOUSNESS', 'cautious', 'skeptical', ...], 'C')]

documents = D_doc + I_doc + S_doc + C_doc
random.shuffle(x=documents) 

N_features =180
all_words = FreqDist(pcr_D.words() + pcr_I.words() + pcr_S.words() + pcr_C.words())   
word_features = list(all_words)[:N_features] 

def document_features(document_words):
    document_words = set(document_words)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words) 
    return features

N_testing = 20
featuresets = [(document_features(d), c) for (d,c) in documents] 
  
train_set, test_set = featuresets[N_testing:], featuresets[:N_testing]

classifier = NaiveBayesClassifier.train(train_set) 
#A ELE probability distribution must have at least one bin.

print(classify.accuracy(classifier, test_set))

print(classifier.show_most_informative_features(5))


'''



#words = [word for word, pos in tagged_words if pos not in ['m']] 
#            words=[]
#            for word, pos in tagged_words:
#                if pos not in ['m']:
#                    words.append(word)

#list=[1, 5, 6, 7]
#newlist=[x**2 for x in list]
#newlist: 1, 25, 36, 49




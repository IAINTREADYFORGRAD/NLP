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

#text 
character=["Hermione", "Harry", "Malfoy", "Dudley"]
char_stopwords=["stopwords_Hermione.txt", "stopwords_Harry.txt", "stopwords_Malfoy.txt", "stopwords_Dudley.txt"]

for i in range (0, 1): 
    stopwords=[]
    line_with_name=[]
    words=[]      
    for stopword in open(file="%s" %char_stopwords[i] , mode="r", encoding="utf8"):
        stopwords.append(stopword.strip())
    with open(file="harry_potter.txt", mode="r", encoding="utf8") as file1:
        text=file1.read().replace("\n", " ")
        cutsentence=re.split(r'[\n;.?!]', text) 
        for sentence in cutsentence:
            b=re.search(pattern="“" + ".*" + character[i]+ ".*" + "”", string=sentence)
            a=re.search(pattern=character[i], string=sentence)
            if a != b:
                line_with_name.append(sentence)
    with open(file="%s.txt" %character[i], mode="w", encoding="utf8") as file2:
        for sentence in line_with_name:
            cutword=psg.cut(sentence)
            for word in cutword:
                if word.word not in stopwords:
                    words.append(word.word)
        file2.write(" ".join(words).strip())


#DISC classifier
stopwords=[]
for stopword in open(file="stopwords_DISC.txt", mode="r", encoding="utf8"):
    stopwords.append(stopword.strip()) 

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
                file2.write(" ".join(words).strip()) 
                    
N_features =100

#D
c="Dok"
pcr_D = PlaintextCorpusReader(root=c, fileids=".*\.txt") 
D_doc = [(pcr_D.words(fileid),c) for fileid in pcr_D.fileids()]
#ALL=FreqDist(pcr_D.words())
#print(list(ALL)[:N_features])

#I
c="Iok"
pcr_I = PlaintextCorpusReader(root=c, fileids=".*\.txt") 
I_doc = [(pcr_I.words(fileid),c) for fileid in pcr_I.fileids()]
#ALL=FreqDist(pcr_I.words())
#print(list(ALL)[:N_features])

#S
c="Sok"
pcr_S = PlaintextCorpusReader(root=c, fileids=".*\.txt") 
S_doc = [(pcr_S.words(fileid),c) for fileid in pcr_S.fileids()]
#ALL=FreqDist(pcr_S.words())
#print(list(ALL)[:N_features])

#C
c="Cok"
pcr_C = PlaintextCorpusReader(root=c, fileids=".*\.txt") 
C_doc = [(pcr_C.words(fileid),c) for fileid in pcr_C.fileids()]
#ALL=FreqDist(pcr_S.words())
#print(list(ALL)[:N_features])


documents = D_doc + I_doc + S_doc + C_doc
random.shuffle(x=documents) 

def document_features(document_words):
    document_words = set(document_words)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words) 
    return features

N_features =100
all_words = FreqDist(pcr_D.words() + pcr_I.words() + pcr_S.words() + pcr_C.words())   
word_features = list(all_words)[:N_features] 

N_testing = 20
featuresets = [(document_features(d), c) for (d,c) in documents] 

#machine training
train_set, test_set = featuresets[N_testing:], featuresets[:N_testing]
classifier = NaiveBayesClassifier.train(train_set) 

print(classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(10))


#testing & predict
c="c10"
pcr_c10 = PlaintextCorpusReader(root=".\c", fileids="c10.txt")
c10_doc = [(pcr_c10.words(fileid),c) for fileid in pcr_c10.fileids()]
c10_features=[(document_features(d), c) for (d,c) in c10_doc] 
for a, b in c10_features:
    result=classifier.classify(a)
print(result) #C

c="s10"
pcr_s10 = PlaintextCorpusReader(root=".\S", fileids="s10.txt")
s10_doc = [(pcr_s10.words(fileid),c) for fileid in pcr_s10.fileids()]
s10_features=[(document_features(d), c) for (d,c) in s10_doc] 
for a, b in s10_features:
    result=classifier.classify(a)
print(result) #S

c="c30"
pcr_c30 = PlaintextCorpusReader(root=".\C", fileids="c30.txt")
c30_doc = [(pcr_c30.words(fileid),c) for fileid in pcr_c30.fileids()]
c30_features=[(document_features(d), c) for (d,c) in c30_doc] 
for a, b in c30_features:
    result=classifier.classify(a)
print(result) #C

c="I35"
pcr_i35 = PlaintextCorpusReader(root=".\I", fileids="I35.txt")
i35_doc = [(pcr_i35.words(fileid),c) for fileid in pcr_i35.fileids()]
i35_features=[(document_features(d), c) for (d,c) in i35_doc] 
for a, b in i35_features:
    result=classifier.classify(a)
print(result) #I

c="I25"
pcr_i25 = PlaintextCorpusReader(root=".\I", fileids="I25.txt")
i25_doc = [(pcr_i25.words(fileid),c) for fileid in pcr_i25.fileids()]
i25_features=[(document_features(d), c) for (d,c) in i25_doc] 
for a, b in i25_features:
    result=classifier.classify(a)
    result=classifier.classify(a)
print(result) #I

c="D15"
pcr_d15 = PlaintextCorpusReader(root=".\D", fileids="D15.txt")
d15_doc = [(pcr_d15.words(fileid),c) for fileid in pcr_d15.fileids()]
d15_features=[(document_features(d), c) for (d,c) in d15_doc] 
for a, b in d15_features:
    result=classifier.classify(a)
print(result) #D

c="Hermione"
pcr_her = PlaintextCorpusReader(root="./", fileids="Hermione.txt")
her_doc = [(pcr_her.words(fileid),c) for fileid in pcr_her.fileids()]
her_features=[(document_features(d), c) for (d,c) in her_doc] 
for a, b in her_features:
    result=classifier.classify(a)
print("Hermione: %s" %result) #C

c="Harry"
pcr_ha = PlaintextCorpusReader(root="./", fileids="Harry.txt")
ha_doc = [(pcr_ha.words(fileid),c) for fileid in pcr_ha.fileids()]
ha_features=[(document_features(d), c) for (d,c) in ha_doc] 
for a, b in ha_features:
    result=classifier.classify(a)
print("Harry: %s" %result) #S

c="Malfoy"
pcr_mal = PlaintextCorpusReader(root="./", fileids="Malfoy.txt")
mal_doc = [(pcr_mal.words(fileid),c) for fileid in pcr_mal.fileids()]
mal_features=[(document_features(d), c) for (d,c) in mal_doc] 
for a, b in mal_features:
    result=classifier.classify(a)
print("Malfoy: %s" %result) #I

c="Dudley"
pcr_dud = PlaintextCorpusReader(root="./", fileids="Dudley.txt")
dud_doc = [(pcr_dud.words(fileid),c) for fileid in pcr_dud.fileids()]
dud_features=[(document_features(d), c) for (d,c) in dud_doc] 
for a, b in dud_features:
    result=classifier.classify(a)
print("Dudley: %s" %result) #I






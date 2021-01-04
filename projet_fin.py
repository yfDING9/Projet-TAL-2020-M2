# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 21:59:32 2020

@author: 64584
"""

from gensim.models import KeyedVectors
import xml.etree.ElementTree as ET
import spacy
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
from collections import Counter
from sklearn.metrics import accuracy_score

# charger le modèle
nlp = spacy.load("fr_core_news_sm")

model = KeyedVectors.load_word2vec_format("frWiki_no_lem_no_postag_no_phrase_1000_skip_cut100.bin", binary=True, unicode_errors="ignore")

#parser XML et recupérer des données
tree = ET.parse('t2.xml')
root = tree.getroot()

source = []
for a in root.findall(".//ensemble/source"):
    source.append(a.text)


cible = []
for a in root.findall(".//ensemble/cible"):
    cible.append(a.text)
    
correct = []
for a in root.findall(".//ensemble"):
    correct.append(a.attrib["cible"])

#supprimer les ponctuations
def remove(liste):
    clean = []
    for phrase in liste :
        phrase = re.sub(r"\d*|\.|,|«|»|\/|\'|\!|\?|:|;","",phrase)
        phrase = phrase.replace("[","")
        phrase = phrase.replace("]","")
        phrase = phrase.replace("(","")
        phrase = phrase.replace("]","")
        phrase = phrase.replace(")","")
        clean.append(phrase)
        
    return clean

#tokenisation
def tokenisation (liste):
    liste_nlp = []
    for item in liste :
        item = nlp(item)
        liste_nlp.append(item)
    
    return liste_nlp

#générer une liste de chaque phrase       
def phrase(liste):
    alll = []
    for word in liste :
        sentence = []
        for item in word :
            sentence.append(item.text)
            
        alll.append(sentence)
            
    return alll

#vectorisation et distance cosinus
def vec_cos(source,cible):
    source_nlp = tokenisation(source)
    cible_nlp = tokenisation(cible)
    phrase_source = phrase(source_nlp)
    phrase_cible = phrase(cible_nlp)
    f = open("res.txt","w",encoding = "utf-8")
        
    vec_source = []
    vec_cible = []
    
    for item in phrase_source :
        temp = []
        for word in item :
            try :
                word = model[word]
                temp.append(word)
      
            except KeyError :
                word = np.zeros(1000)
                temp.append(word)
                    
        vec_source.append(np.mean(temp,axis = 0))
        
    for item in phrase_cible :
        temp = []
        for word in item :
            try :
                word = model[word]
                temp.append(word)

            except KeyError :
                word = np.zeros(1000)
                temp.append(word)

        vec_cible.append(np.mean(temp,axis = 0))
        
    vec1_cible = vec_cible[:]
    
    total = []
    
    for array in vec_source :
        a = []
        a.append(array)
        f.write(str(a))
        
        for i in range(0,3):
            c = []
            c.append(vec1_cible.pop(0))
            for cos_array in cosine_similarity(a,c) :
                for cos in cos_array :
                    total.append(cos)
                    
  
    total1 = total[:]
    res_cos = []
    for item in source :
        compare = []
        for i in range (0,3):
            compare.append(total1.pop(0))
        res_cos.append(str(compare.index(max(compare))+1))
        
        
    print(accuracy_score(correct, res_cos))
    
    
    return res_cos
    
# distance jaccard
def jaccard(source_clean,cible_clean):
    cible3 = cible_clean[:]
    res = []
    for item in source_clean :
        compare = []
        compare.append(item)
        mots = []
        jac = []
        for i in range(0,3):
            compare.append(cible3.pop(0))
        
        for phrase in compare :
            mots.append(phrase.split(" "))
        
        
        for k in range(1,4):
            temp = 0
            for word in mots[0]:
                if word in mots[k]:
                    temp = temp + 1
                    
            num = len(mots[0])+len(mots[k])-temp
            coef = float(temp/num)
            jac.append(coef)
            
        result = Counter(jac)
        for (key,value) in result.items():
                if key == max(jac):
                    res.append(str(jac.index(key)+1))
                
    print(accuracy_score(correct, res))
   
 #distance Levenshtein
def lev(source,cible):
    cible2 = cible[:]
    res_lev = []
    for a in source :
        compare = []
        for i in range(0,3):
            dis = Levenshtein.distance(a,cible2.pop(0))
            compare.append(dis)
                        
        result = Counter(compare)
        for (key,value) in result.items():
            if key == min(compare):
                res_lev.append(str(compare.index(key)+1))
                    
                    
        
            
    print(accuracy_score(correct, res_lev))
 
#lancement de fonction  
source_clean = remove(source)
cible_clean = remove(cible)
cosine = vec_cos(source_clean,cible_clean)
jaccard(source_clean,cible_clean)
lev(source_clean,cible_clean)

#préciser des erreurs
for i,item in enumerate(cosine) :
    if item != correct[i]:
        print(i+1)
    




            
            
            
            
            
            
        
                

    
           
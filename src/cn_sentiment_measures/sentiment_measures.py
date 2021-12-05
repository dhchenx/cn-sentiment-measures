import pickle
import jieba
import re
import math
import os
# https://stackoverflow.com/questions/33543446/what-is-the-formula-of-sentiment-calculation
class SentimentMeasures:
    def __init__(self,root_path=None):
        if root_path==None:
            root_path = os.path.dirname(os.path.realpath(__file__))

        self.lib_negative_words = pickle.load(open(os.path.join(root_path,"libs","dict_negative_words.pickle"), 'rb'))
        self.lib_positive_words = pickle.load(open(os.path.join(root_path,"libs","dict_positive_words.pickle"), 'rb'))
        self.lib_degree_words = pickle.load(open(os.path.join(root_path,"libs","dict_degree_words.pickle"), 'rb'))
        for w in self.lib_negative_words:
            jieba.add_word(w)
        for w in self.lib_positive_words:
            jieba.add_word(w)
        for k in self.lib_degree_words.keys():
            for w in self.lib_degree_words[k]:
                jieba.add_word(w)
        self.stop_words = open(os.path.join(root_path,"libs","stopwords-zh.txt"), 'r', encoding='utf-8').readlines()
        self.stop_words = [w.strip() for w in self.stop_words]
        self.word_score_function=None

    def word_segmentation(self,str):
        str = re.sub(r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", str)
        seg_list = jieba.cut(str, cut_all=False)
        r_list = []
        for w in seg_list:
            if w not in self.stop_words:
                if w.strip() == "":
                    continue
                r_list.append(w.strip())
        return r_list

    def add_word_list(self,wlist):
        for w in wlist:
            jieba.add_word(w)

    def set_word_score_function(self,func):
        self.word_score_function=func

    def get_word_score(self,word,type):
        if self.word_score_function!=None:
            return self.word_score_function(word,type)
        else:
            return 1

    def get_degree_score(self,word):
        keys=["extreme","very","more","over","ish","insufficiently"]
        key_scores=[2,1.75,1.5,1.25,1.00,0.5]
        for idx,k in enumerate(keys):
            if word in self.lib_degree_words[k]:
                return key_scores[idx]
        return None

    def is_positive_or_negative(self,w):
        if w in self.lib_positive_words or w in self.lib_negative_words:
            return True
        else:
            return False

    # get proportional negative or positive words
    def NPPercent(self,word_list):
        P = 0
        N = 0
        O = 0
        for w in word_list:
            if w in self.lib_negative_words:
                N += self.get_word_score(w,"N")
            elif w in self.lib_positive_words:
                P += self.get_word_score(w,"P")
                continue
            else:
                O += self.get_word_score(w,"O")
        return (P+N)/(P+N+O)

    # Absolute Proportional Difference. Bounds: [0,1]
    def APD(self,word_list):
        P = 0
        N = 0
        O = 0
        for w in word_list:
            if w in self.lib_negative_words:
                N += self.get_word_score(w,"N")
            elif w in self.lib_positive_words:
                P += self.get_word_score(w,"P")
                continue
            else:
                O += self.get_word_score(w,"O")
        return (P - N) / (P + N + O)

    # Relative Proportional Difference. Bounds: [-1, 1]
    def RPD(self, word_list):
        P = 0
        N = 0
        O = 0
        for w in word_list:
            if w in self.lib_negative_words:
                N += self.get_word_score(w,"N")
            elif w in self.lib_positive_words:
                P += self.get_word_score(w,"P")
                continue
            else:
                O += self.get_word_score(w,"O")
        if P+N==0:
            return 0
        return (P - N) / (P + N)

    # Relative Proportional Difference with degree. Bounds: [-1, 1]
    def RPD_with_degree(self, word_list):
        P = 0
        N = 0
        O = 0
        for idx,w in enumerate(word_list):
            if w in self.lib_negative_words:
                if idx > 0:
                    degree_score = self.get_degree_score(word_list[idx - 1])
                    if degree_score != None:
                        N += self.get_word_score(w,"N") * degree_score
                    else:
                        N += self.get_word_score(w,"N")
            elif w in self.lib_positive_words:
                if idx > 0:
                    degree_score = self.get_degree_score(word_list[idx - 1])
                    if degree_score != None:
                        P += self.get_word_score(w,"P")* degree_score
                    else:
                        P += self.get_word_score(w,"P")
            else:
                O += self.get_word_score(w,"O")
        if P+N==0:
            return 0
        return (P - N) / (P + N)

    # Logit scale. Bounds: [-infinity, +infinity]
    def LS(self, word_list):
        P = 0
        N = 0
        O = 0
        for w in word_list:
            if w in self.lib_negative_words:
                N += self.get_word_score(w,"N")
            elif w in self.lib_positive_words:
                P += self.get_word_score(w,"P")
            else:
                O += self.get_word_score(w,"O")

        P_adjusted=P+0.5
        N_adjusted=N+0.5
        if P_adjusted<=0 or N_adjusted<=0:
            return None

        return math.log10(P_adjusted)-math.log10(N_adjusted)

    # Logit scale with degree. Bounds: [-infinity, +infinity]
    def LS_with_degree(self, word_list):
        P = 0
        N = 0
        O = 0
        for idx,w in enumerate(word_list):
            if w in self.lib_negative_words:
                if idx > 0:
                    degree_score = self.get_degree_score(word_list[idx - 1])
                    if degree_score != None:
                        N += self.get_word_score(w,"N") * degree_score
                    else:
                        N += self.get_word_score(w,"N")
            elif w in self.lib_positive_words:
                if idx > 0:
                    degree_score = self.get_degree_score(word_list[idx - 1])
                    if degree_score != None:
                        P += self.get_word_score(w,"P") * degree_score
                    else:
                        P += self.get_word_score(w,"P")
            else:
                O += self.get_word_score(w,"O")

        P_adjusted = P + 0.5
        N_adjusted = N + 0.5
        if P_adjusted <= 0 or N_adjusted <= 0:
            return None

        return math.log10(P_adjusted) - math.log10(N_adjusted)

    # Absolute Proportional Difference with degree. Bounds: [0,1]
    def APD_with_degree(self, word_list):
        P = 0
        N = 0
        O = 0
        for idx,w in enumerate(word_list):
            if w in self.lib_negative_words:
                if idx>0:
                    degree_score=self.get_degree_score(word_list[idx-1])
                    if degree_score != None:
                        N+=self.get_word_score(w,"N")*degree_score
                    else:
                        N+=self.get_word_score(w,"N")
            elif w in self.lib_positive_words:
                if idx > 0:
                    degree_score = self.get_degree_score(word_list[idx - 1])
                    if degree_score != None:
                        P += self.get_word_score(w,"P") * degree_score
                    else:
                        P += self.get_word_score(w,"P")
            else:
                O += self.get_word_score(w,"O")
        Score=(P - N) / (P + N + O)

        if P+N+O==0:
            return None

        if Score>1:
            Score=1
        elif Score<-1:
            Score=-1
        return Score

from statistics import mean
import os
import sys
import re as re
from enchant.tokenize import tokenize
from nltk.sem.logic import ExpectedMoreTokensException
from oauthlib.oauth2.rfc6749.grant_types.authorization_code import code_challenge_method_plain
from scipy.optimize._tstutils import f1

#sys.path.insert(0, "D:/program files/scitools/bin/pc-win64/python")
#os.add_dll_directory('D:/Program Files/SciTools/bin/pc-win64')
os.add_dll_directory(r"D:\Program Files\SciTools\bin\pc-win64\Python")
#import understand
#understand.version()
#import understand as und

#import understand
from tkinter import *

import time
import tkinter as tk
from os import listdir
from os.path import isfile, join
import glob, os
import numpy as np
import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
#pd.set_option('display.max_columns', None)
#pd.set_option('max_columns', None)
#pd.set_option('max_colwidth', None)
#pd.set_option('max_rows', None)
#pd.set_option('max_seq_item', None)

import tkinter
import threading
import csv
import math
import metrics_names
from naming import UnderstandUtility
from Metrics import cls_get_metrics as JCodeOdorMetric
#import code2vec  as code
import warnings
import joblib
from datetime import datetime

from tkinter import filedialog
from numpy import genfromtxt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
import pickle
import random
import nltk
from nltk.corpus import wordnet
#nltk.download()
import enchant
eng_dict = enchant.Dict("en_US")

class Calculate_Metrics:

    """Initiate Main Form"""


    def main(self):
        win = tk.Tk()
        win.title('Code Smell Detector')
        win.geometry('600x100')
        setPathbtn = tk.Button(win, text='SetPath', command=lambda: self.search_for_file_path(win))
        setPathbtn.pack(side=tk.LEFT)
        ExtractUdbsbtn = tk.Button(win, text='ExtractUdbs', command=self.create_understand_database_from_project)
        ExtractUdbsbtn.pack(side=tk.LEFT)
        calcmetricbtn = tk.Button(win, text='CalcMetrics', command=self.Save_Metrics)
        calcmetricbtn.pack(side=tk.LEFT)
        calcThresholdbtn = tk.Button(win, text='CalcThresholds', command=self.Save_Thresholds)
        calcThresholdbtn.pack(side=tk.LEFT)
        calccodesmelldbtn = tk.Button(win, text='CalcCodesmell', command=self.Calc_Codesmell)
        calccodesmelldbtn.pack(side=tk.LEFT)
        calcmetricsbtn = tk.Button(win, text='CalcMetrics', command=self.Metrics)
        calcmetricsbtn.pack(side=tk.LEFT)
        Learnbtn = tk.Button(win, text='Train', command=self.StartLearn)
        Learnbtn.pack(side=tk.LEFT)
        Testbtn = tk.Button(win, text='Test', command=self.StartTest)
        Testbtn.pack(side=tk.LEFT)
        DRParsaTheoryBTN = tk.Button(win, text='DRParsaTheory', command=self.DRParsaTheory)
        DRParsaTheoryBTN.pack(side=tk.LEFT)
       # Evaluatebtn = tk.Button(win, text='Evaluate', command=self.StartEvaluate)
       # Evaluatebtn.pack(side=tk.LEFT)
        win.mainloop()

    def StartLearn(self):
        t1 = threading.Thread(target=self.Learn)
        t1.start()

    def StartTest(self):
        t1 = threading.Thread(target=self.Test)
        t1.start()


    def DRParsaTheory(self):
        t1 = threading.Thread(target=self.DRParsa)
        t1.start()

    #def StartEvaluate(self):
     #   t1 = threading.Thread(target=self.Test)
     #   t1.start()

    def StartCalcMetrics(self):
        t1 = threading.Thread(target=self.Metrics)
        t1.start()

    """Initiate A filedialog To Get Projects Directory"""
    def search_for_file_path(self,root):
        currdir = os.getcwd()
        tempdir = filedialog.askdirectory(parent=root, initialdir=r'C:\Users\moham\Desktop\java', title='Please Select A Udbs Directories')
        rootpath = tempdir + "/"
        os.chdir(rootpath)
        return   rootpath

    """Save Metrics TO SCV"""
    def Metrics(self):
        #print("Start Process at " + str(datetime.now()))
        root_path = os.getcwd()
        root_path = root_path + '\\'
        root_path = root_path.replace('\\', '/')
        self.extract_metrics_and_coverage_all(root_path,None,root_path)

    """Fetch Data From CSV And Begin Learn Process"""
    def Learn(self):
        #print("Start Process at " + str(datetime.now()))
        root_path = os.getcwd()
        root_path = root_path + '\\'
        root_path = root_path.replace('\\', '/')
        frame = pd.DataFrame()
        if not os.path.isfile(os.path.join(root_path, 'Final.csv')):
            print('Begin Create Final Csv File')
            allFiles = glob.glob(root_path + "/*.csv")
            list_ = []
            for file_ in allFiles:
              df = pd.read_csv(file_,index_col=None, header=None, low_memory=False)
              re = df.iloc[1:, :]
              re = re.iloc[:, :].replace([None], np.nan)
              re = re.iloc[:, :].replace('None', np.nan,)
              re = re.iloc[:, :].replace('NaN', np.nan)
              re = re[re[3].notna()]
              list_.append(re)
            frame = pd.concat(list_)

            frame.iloc[:, 4:] = Normalizer().fit_transform(frame.iloc[:, 4:].replace(np.nan, 0))

            frame.to_csv(root_path+'/Final.csv', header=False,index=False)
        else :

            frame = pd.read_csv(root_path+'Final.csv',header=None, low_memory=False)

        X, Y = frame.iloc[:, 4:], frame.iloc[:, :4]

        nneighbors=10

        neigh = NearestNeighbors(n_neighbors=nneighbors,algorithm='auto',metric='minkowski',n_jobs=3).fit(X)

        if not os.path.isfile(os.path.join(root_path, 'pickle.pkl')):
            print('Begin Create pickle pkl File')
            with open("pickle.pkl", 'wb') as pickleFile:
             pickle.dump(neigh, pickleFile)
             pickleFile.close()

        with open("pickle.pkl", 'rb') as pickleFile:
            neigh = pickle.load(pickleFile)


        n = random.randint(0, len(X.index))
        #n=5056

        #n=28237
        #print(n)
        sample = X.iloc[[n],:]
        #print(sample.values)
        #print('')
        print('-----------------------------Sample---------------------------------')
        print('')
        print(str(n) + ' - ' + Y.iloc[[n],:].values[0,3])
        print('')
        #stokens = nltk.word_tokenize(Y.iloc[[n],:].values[0,3])
        #schars = [self.segment_str(chars) for chars in stokens][0]
        #print(schars)
        #print('')
        #print(X.iloc[[n], :].values)
        #print(sample)
        #distances, indices = neigh.kneighbors(sample,10,return_distance=True)
        distances, indices = neigh.kneighbors(sample, nneighbors, return_distance=True)
        #print(distances)
        #print(indices)
        #sparr = np.char.split(indices)
        #print(sparr)
        #print(indices)

        #zerolist = list()
        #i =0
        #for i in  range(0,nneighbors-1) :
         #i=i+1
         #if distances[0,i]==0 :
             #zerolist.append(indices[0, i])
         #print(indices[0, i])
        #print(zerolist)
        #for a in zerolist :
            #print(a)
            #print(X[a])
            #print(X.iloc[[a], :])
            #ylist = pd.Series( Y.iloc[[a], :].values[0])
            #print(frame)
            #listmethods = frame.loc[frame['1']==ylist[0]]
            #print(listmethods)

        #return
        #distances = np.array(distances)
        #indices = np.array(indices)
        #matrix = np.concatenate(indices,distances)
        #indices=indices.tolist()
        #distances = distances.tolist()


        #print (df)
        #print(indices.shape[0])
        #print(indices.shape[1])

        #print(distances[(distances=0])
        #print([Y.iloc[indices[row, col], :] for row in range(indices.shape[0]) for col in range(indices.shape[1])])
        fram = pd.read_csv(root_path + 'Final1.csv', header=None, low_memory=False)
        X1, Y1 = fram.iloc[:, 4:], fram.iloc[:, :4]
        #print(Y)

        print('-------------------------------indices & distances--------------------------------------------------------------------')
        print('')
        print(indices)
        print(distances)
        print('')
        print('---------------------------------------------------------------------------------------------------------------------')
        df = pd.DataFrame(
            [indices[row, col] for row in range(indices.shape[0]) for col in range(indices.shape[1])])
        for y in df.values :
            print('')
            #print(y[0])
            ylist =  Y1.iloc[[y[0]], :].values[0,3]
            print(str(y[0]) + ' - ' +ylist)

            #tokens = nltk.word_tokenize(ylist)
            #chars = [self.segment_str(chars) for chars in tokens][0]
            #print('')
            #print(chars)
            print('')
            print('**********************************')

            #print(tokens)
            #print(df.values[0])
        #print(Y.iloc[indices[row,col],:])
        #print(Y.iloc[[149], :].values)
        #print(Y.iloc[[151], :].values)
        #print(Y.iloc[[148], :].values)
        #print(Y.iloc[[152], :].values)
        #print(Y.iloc[[94057], :].values)
        #print(Y.iloc[[111572], :].values)
        #print(Y.iloc[[111572], :].values)
        #print(Y.iloc[[15814], :])
        #print(Y.iloc[[64413], :])
        #print(Y.head(10))
        print('---------------------------------------------------------------------------------------------------------------------')

        print('')
        #print("End Process at " + str(datetime.now()))

    def segment_str(self, chars, exclude=None):
            """
            Segment a string of chars using the pyenchant vocabulary.
            Keeps longest possible words that account for all characters,
            and returns list of segmented words.

            :param chars: (str) The character string to segment.
            :param exclude: (set) A set of string to exclude from consideration.
                            (These have been found previously to lead to dead ends.)
                            If an excluded word occurs later in the string, this
                            function will fail.
            """
            words = []

            if not chars.isalpha():  # don't check punctuation etc.; needs more work
                return [chars]

            if not exclude:
                exclude = set()

            working_chars = chars
            while working_chars:
                # iterate through segments of the chars starting with the longest segment possible
                for i in range(len(working_chars), 1, -1):
                    segment = working_chars[:i]
                    if eng_dict.check(segment) and segment not in exclude:
                        words.append(segment)
                        working_chars = working_chars[i:]
                        break
                else:  # no matching segments were found
                    if words:
                        exclude.add(words[-1])
                        return segment_str(chars, exclude=exclude)
                    # let the user know a word was missing from the dictionary,
                    # but keep the word
                    #print('"{chars}" not in dictionary (so just keeping as one segment)!'
                    #      .format(chars=chars))
                    return [chars]
            # return a list of words based on the segmentation
            return words

    """Fetch Data From CSV And Begin Learn Process"""
    def DRParsa(self):
        root_path = os.getcwd()
        root_path = root_path + '\\'
        root_path = root_path.replace('\\', '/')
        frame = pd.DataFrame()
        if not os.path.isfile(os.path.join(root_path, 'FinalCLS.csv')):
            print('Begin Create Final Csv File')
            allFiles = glob.glob(root_path + "/*.csv")
            list_ = []
            for file_ in allFiles:
              print(file_)
              df = pd.read_csv(file_,index_col=None,  low_memory=False)
              re = df.iloc[1:, :]
              re = re.iloc[:, :].replace([None], np.nan)
              re = re.iloc[:, :].replace('None', np.nan,)
              re = re.iloc[:, :].replace('NaN', np.nan)
              re.iloc[:, 4:].replace(np.nan,0)
              #re.iloc[:, 4:] = Normalizer().fit_transform(re.iloc[:, 4:].replace(np.nan, 0))
              #print(re.iloc[:, 4:])
              #re.to_csv(root_path + '/FinalCLS1.csv', header=False, index=False)
              grouped_df=re.groupby(['Project','Package','Class'], as_index=False)
              mean_df = grouped_df.mean()
              #print(mean_df.iloc[:, :3])
              #print(mean_df.iloc[:, 3:])
              #mean_df.to_csv(root_path + '/FinalCLS2.csv', header=False, index=False)
              #return
              #re = re[re[3].notna()]
              #print(re)
              #return
              list_.append(mean_df)

            frame = pd.concat(list_)

            #print(frame)
            #print('start')
            frame.iloc[:, 4:] = Normalizer().fit_transform(frame.iloc[:, 4:].replace(np.nan, 0))
           # print('Here')
            #print(frame.reset_index())
            #print(frame[3])
            #print(frame.groupby(frame[3]))
            #return
            #print('mid')
            frame.to_csv(root_path+'/FinalCLS.csv', header=False,index=False)
            #print('after')
            #return
        else :

            frame = pd.read_csv(root_path+'FinalCLS.csv',header=None, low_memory=False)

        X, Y = frame.iloc[:, 3:], frame.iloc[:, :3]

        nneighbors=10
        #print(X)
        #print(Y)
        neigh = NearestNeighbors(n_neighbors=nneighbors,algorithm='auto',metric='minkowski',n_jobs=3).fit(X)

        if not os.path.isfile(os.path.join(root_path, 'pickleCLS.pkl')):
            print('Begin Create pickle pkl CLS File')
            with open("pickleCLS.pkl", 'wb') as pickleFile:
             pickle.dump(neigh, pickleFile)
             pickleFile.close()

        with open("pickleCLS.pkl", 'rb') as pickleFile:
            #print('pickle pkl File Loaded')
            neigh = pickle.load(pickleFile)

        def segment_str(chars, exclude=None):
            """
            Segment a string of chars using the pyenchant vocabulary.
            Keeps longest possible words that account for all characters,
            and returns list of segmented words.

            :param chars: (str) The character string to segment.
            :param exclude: (set) A set of string to exclude from consideration.
                            (These have been found previously to lead to dead ends.)
                            If an excluded word occurs later in the string, this
                            function will fail.
            """
            words = []

            if not chars.isalpha():  # don't check punctuation etc.; needs more work
                return [chars]

            if not exclude:
                exclude = set()

            working_chars = chars
            while working_chars:
                # iterate through segments of the chars starting with the longest segment possible
                for i in range(len(working_chars), 1, -1):
                    segment = working_chars[:i]
                    if eng_dict.check(segment) and segment not in exclude:
                        words.append(segment)
                        working_chars = working_chars[i:]
                        break
                else:  # no matching segments were found
                    if words:
                        exclude.add(words[-1])
                        return segment_str(chars, exclude=exclude)
                    # let the user know a word was missing from the dictionary,
                    # but keep the word
                    print('"{chars}" not in dictionary (so just keeping as one segment)!'
                          .format(chars=chars))
                    return [chars]
            # return a list of words based on the segmentation
            return words

        n = random.randint(0, len(X.index))
        #n=5056

        #n=28237
        #print(n)
        sample = X.iloc[[n],:]

        print('-----------------------------Main Test Sample---------------------------------')
        print('')
        name = Y.iloc[[n],:].values[0,2];
        lastname = name.split('.')[-1]
        print('FullName   ->  ' +str(n) + ' - ' + name)
        print('ClassName  ->  ' +lastname)
        tokens = nltk.word_tokenize(lastname)
        chars = [segment_str(chars) for chars in tokens][0]
        print('Tokens     ->  ' ,chars)
        print('')

        distances, indices = neigh.kneighbors(sample, nneighbors, return_distance=True)

        fram = pd.read_csv(root_path + 'FinalCLS1.csv', header=None, low_memory=False)
        X1, Y1 = fram.iloc[:, 3:], fram.iloc[:, :3]

        print('-------------------------------indices & distances--------------------------------------------------------------------')
        print('')
        print('Indices       ->  ' , indices)
        print('Distances     ->  ' , distances)
        print('')
        print('---------------------------------------------------------------------------------------------------------------------')
        df = pd.DataFrame(
            [indices[row, col] for row in range(indices.shape[0]) for col in range(indices.shape[1])])
        count = 1
        finallist1 = list()
        finallist2 = list()
        for y in df.values :
            print('')
            print('****************************************  Sample : '+ str(count) + ' ***************************************')
            print('')
            ylist =  Y1.iloc[[y[0]], :].values[0,2]
            lastlist = ylist.split('.')[-1]
            print('FullName   ->  ' +str(y[0]) + ' - ' +ylist)
            print('ClassName   ->  ' +lastlist)
            tokens1 = nltk.word_tokenize(lastlist)
            chars1 = [segment_str(chars1) for chars1 in tokens1][0]
            print('Tokens     ->  ', chars1)
            count = count+1
            print('')
            llist=list()
            for f in chars :
              flist=list()
              for j in chars1 :
                wp1 = wordnet.synsets(f)
                wp2 = wordnet.synsets(j)
                if wp2 and wp1:  # Thanks to @alexis' note
                    s = wp1[0].wup_similarity(wp2[0])
                    flist.append(0 if s is None else s)
                    print('Similarity Score With "'+str(f)+'" And "'+ str(j)+'" Is : '+ str(s))
              if len(flist) :
               print(flist)
               print('Max Score With "'+str(f) + ' Is :' + str(max(flist)))
               llist.append(max(flist))
               print('')
            print('')
            print(llist)
            print('Average Score :' + str((0 if len(llist) == 0 else sum(llist) / len(llist)) ))
            finallist1.append(lastlist)
            finallist2.append((0 if len(llist) == 0 else sum(llist) / len(llist)) )

        print('---------------------------------------------------Sorted Final Result------------------------------------------------------------------')
        data = {'Name': finallist1,
                'Score': finallist2}
        print('')
        print('Main Example : ' + lastname)
        print('')
        df = pd.DataFrame(data)
        print(df.sort_values('Score', ascending=False))
        print('')


    def segment_str(self,inputchars):
            words = []
            working_chars = inputchars
            for i in range(len(working_chars), 1, -1):
                    segment = working_chars[:i]
                    if eng_dict.check(segment):
                        words.append(segment)
                        working_chars = working_chars[i:]
                        break
            if working_chars:
                        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', working_chars)
                        camel_cases = [m.group(0) for m in matches]
                        for case in camel_cases:
                            case = case.lower()
                            case = case.split('_')
                            words.extend(case)
            return words

    """Fetch Data From CSV And Begin Learn Process"""
    def Test(self):
        pd.set_option('display.max_rows', None, 'display.max_columns', None)  # more options can be specified also
        root_path = os.getcwd()
        root_path = root_path + '\\'
        root_path = root_path.replace('\\', '/')

        TestDataframe = pd.DataFrame()
        frames = []
        if not os.path.isfile(os.path.join(root_path, 'FinalTest.pkl')):
            allTestClsFiles = glob.glob(root_path + "/*.csv")

            for Clsfile in allTestClsFiles:
                LoadClsFile = pd.read_csv(Clsfile, index_col=None, header=None, low_memory=False)
                re = LoadClsFile.iloc[1:, :]
                re = re.iloc[:, :].replace([None], np.nan)
                re = re.iloc[:, :].replace('None', np.nan)
                re = re.iloc[:, :].replace('NaN', np.nan)
                re = re[re[3].notna()]
                re.iloc[:, 4:] = Normalizer().fit_transform(re.iloc[:, 4:].replace(np.nan, 0))
                frames.append(re)

            TestDataframe = pd.concat(frames)
            TestDataframe.to_pickle('FinalTest.pkl')
        else:
            TestDataframe = pd.read_pickle('FinalTest.pkl')

        #print(len(TestDataframe.index))
        TestX, TestY = TestDataframe.iloc[:, 4:], TestDataframe.iloc[:, :4]
        nneighbors = 10
        AlgorithmList = ['auto']
        MetricList = ['minkowski', 'euclidean', 'manhattan', 'chebyshev']

        def dot(A, B):
            return (sum(a * b for a, b in zip(A, B)))

        def cosine_similarity(a, b):
            a1 = dot(a, b)
            a2 = (dot(a, a) ** .5)
            a3 = (dot(b, b) ** .5)
            distance =  a1/ (a2 * a3)
            return 1-distance

        def jaccard(list1, list2):
            intersection = len(list(set(list1).intersection(list2)))
            union = (len(list1) + len(list2)) - intersection
            return float(intersection) / union



        for Algorithm in AlgorithmList:
            for Metric in MetricList:
                if not os.path.isfile(os.path.join(root_path + '/FinalResult_'+Algorithm+'_'+Metric+'.csv')):
                       LearnData = pd.read_pickle('FinalLearn.pkl')
                       LearnX, LearnY = LearnData.iloc[:, 4:], LearnData.iloc[:, :4]

                       #print(str(datetime.now()))
                       #print(len(LearnData.index))
                       neigh = NearestNeighbors(n_neighbors=nneighbors, algorithm=Algorithm, metric=Metric,
                                                n_jobs=4).fit(LearnX)
                       ResultHeader = ['ExpectedID','ExpectedName','RecommendedName','RecommendedID','WuPalmerScore','Precision','Recall','F1Score']
                       finalResult = list()
                       finalResult.append(ResultHeader)

                       nlen = range(len(TestX.index))
                       #nlen = range(420526,420528,1)
                       for n in nlen :
                                   #print(str(datetime.now()))
                                   nneighbors = 10
                                   Testsample = TestX.iloc[[n], :]
                                   print (str(n))
                                   ExpectedMethodID = n
                                   ExpectedMethodName = TestY.iloc[[n], :].values[0, 3]
                                   ExpectedMethodNamechars = self.segment_str(ExpectedMethodName)
                                   #print(Testsample)
                                   distances, indices = neigh.kneighbors(Testsample, nneighbors)

                                   #print(ExpectedMethodName)
                                   #loop=2
                                   max=len(LearnY.index)
                                   max=100
                                   while(distances[0][0]==distances[0][len(distances[0])-1] and nneighbors<max):
                                        distances, indices = neigh.kneighbors(Testsample, nneighbors)
                                        rdf = LearnY.iloc[indices[0], :].values[:,3]
                                        #print(distances)
                                        if ExpectedMethodName in rdf :
                                            break
                                        else :
                                            nneighbors = nneighbors * 100
                                            if nneighbors>max:
                                                nneighbors=max
                                            #loop=loop*2

                                   #print(indices[0])
                                   #tempdata = LearnY.iloc[indices[0],:]
                                   #print(tempdata)
                                   #tempdata = tempdata.loc[tempdata[3]==ExpectedMethodName]
                                   #print(str(datetime.now()))
                                   tempdata = pd.DataFrame(
                                       [[indices[row, col],LearnY.iloc[indices[row, col],:][3]] for row in range(indices.shape[0]) for col in
                                        range(indices.shape[1])])
                                   tempdata = tempdata.loc[tempdata[1] == ExpectedMethodName]
                                   #print(str(datetime.now()))
                                   #print(tempdata)
                                   #print(str(datetime.now()))
                                   #print('here')
                                   if len(tempdata.index) > 0:
                                      # print('here')
                                       Recommendeddf = tempdata.head(1)[0].to_numpy()

                                   else:
                                       Recommendeddf = indices[0]
                                   #print('here')
                                   #print(Recommendeddf)
                                   #print(str(datetime.now()))
                                   #print(Recommendeddf)

                                   ListResult = []
                                   for RC in Recommendeddf:
                                     #print(RC[0])
                                     #print(LearnY.iloc[[RC[0]], :].values[0, 3])
                                     #print(TestX.iloc[[n], :].values[0].tolist())
                                     #print(LearnX.iloc[[RC[0]], :].values[0].tolist())
                                     #jaccardresult = jaccard(TestX.iloc[[n], :].values[0].tolist(),LearnX.iloc[[RC[0]], :].values[0].tolist())
                                     #print(jaccardresult)
                                     #if  jaccardresult > 0.5 :
                                       #print(RC)
                                       RecommendedMethodID = RC
                                       RecommendedMethodName = LearnY.iloc[[RC], :].values[0, 3]
                                       RecommendedMethodNamechars = self.segment_str(RecommendedMethodName)
                                       #print(RecommendedMethodName)
                                       CharsScores=[]

                                       for EXchar in ExpectedMethodNamechars:
                                           EXSyn = wordnet.synsets(EXchar)
                                           MaxScopeSimilarityScore=0
                                           for RCChar in RecommendedMethodNamechars:
                                               RCSyn = wordnet.synsets(RCChar)
                                               if EXSyn and RCSyn:
                                                   SimilarityScore = EXSyn[0].wup_similarity(RCSyn[0])
                                                   SimilarityScore = (0 if SimilarityScore is None else SimilarityScore)
                                                   MaxScopeSimilarityScore = (SimilarityScore if SimilarityScore > MaxScopeSimilarityScore  else MaxScopeSimilarityScore)
                                           CharsScores.append(MaxScopeSimilarityScore)

                                       SameWordsCount = len([value for value in ExpectedMethodNamechars if value in RecommendedMethodNamechars])
                                       #print(str(datetime.now()))
                                       Precision = SameWordsCount/len(RecommendedMethodNamechars)
                                       Recall = SameWordsCount/len(ExpectedMethodNamechars)
                                       SumRecallPrecision = Recall+Precision
                                       f1score = ((2*Precision*Recall)/(SumRecallPrecision) if SumRecallPrecision>0 else 0)
                                       WuPalmerScore = (0 if len(CharsScores) == 0 else sum(CharsScores) / len(CharsScores))
                                       ListResult.append([ExpectedMethodID,ExpectedMethodName,RecommendedMethodName,RecommendedMethodID,WuPalmerScore,Precision,Recall,f1score])

                                   df = pd.DataFrame(ListResult)
                                   df = df.sort_values(by=[7, 4], ascending=False)



                                   data = df.loc[df[2]== ExpectedMethodName]
                                   if len(data.index) > 0 :
                                    bestresult = data.head(1).values[0]
                                   else :
                                    bestresult = df.head(1).values[0]
                                   #print(bestresult)
                                   finalResult.append(bestresult)

                       best = pd.DataFrame(finalResult)
                       outputpath = root_path + '/FinalResult_'+Algorithm+'_'+Metric+'.csv'

                       best.to_csv(outputpath, header=False, index=False)




    """Save Udb Project Metrics To Same Directory In TXT Format"""
    def Save_Metrics(self):
        #print("Start Process at " + str(datetime.now()))
        """Initiate A search Dialog To Get Udb Projects Path"""
        obj_get_metrics =All_Metrics.cls_get_metrics()

        """Create Metrics List"""
        NOPA_List=list()
        NOAM_List=list()
        NOMNAMM_List=list()
        LOCNAMM_List=list()
        WOC_List=list()
        WMCNAMM_List=list()
        TCC_List=list()
        ATFD_CLASS_List=list()
        LOC_List = list()
        CYCLO_List=list()
        MAXNESTING_List = list()
        NOLV_List = list()
        ATLD_List = list()
        CC_List = list()
        CM_List = list()
        FANOUT_List = list()
        CINT_List = list()
        CDISP_List = list()
        MaMCL_List = list()
        MeMCL_List = list()
        NMCS_List = list()
        LAA_List = list()
        FDP_List=list()
        ATFDMETHOD_List = list()

        """Get Udb Files From RootPath and Save In File List"""
        file_list=list()
        for file in glob.glob("*.udb"):
            file_list.append(file)

        """Start Open Udb Files And Calculate Metrics"""
        for project in file_list:
          #print(project)
          db=understand.open(project)
          print("\n______________ project : ",project ,"________________")

          print("loading")

          # Calculate Method Metrics
          def classmetrices():

              countclass = len(db.ents("class"))
              count = 1
              for classname in db.ents("class"):
                if (str(classname.library()) != "Standard"):

                    #Print Execution State
                    exec = (str(count) + "/" + str(countclass))
                    count += 1
                    sys.stdout.write('\r wait : ' + exec)

                    #Start Calculate Class Metrics
                    nopa=obj_get_metrics.NOPA(classname)
                    nopa = float(0 if nopa is None else nopa)
                    #if nopa > 0:
                    NOPA_List.append(nopa)

                    # Calculate noam
                    noam=obj_get_metrics.NOAM(classname)
                    noam = float(0 if noam is None else noam)
                    #if noam > 0:
                    NOAM_List.append(noam)

                    # Calculate nomnamm
                    nomnamm=obj_get_metrics.NOMNAMM(classname)
                    nomnamm = float(0 if nomnamm is None else nomnamm)
                    #if nomnamm > 0:
                    NOMNAMM_List.append(nomnamm)

                    # Calculate locnamm
                    locnamm=obj_get_metrics.LOCNAMM(classname)
                    locnamm = float(0 if locnamm is None else locnamm)
                    #if locnamm > 0:
                    LOCNAMM_List.append(locnamm)

                    # Calculate woc
                    woc=obj_get_metrics.WOC(classname)
                    woc = float(0 if woc is None else woc)
                    #if woc > 0:
                    WOC_List.append(woc)

                    # Calculate wmcnamm
                    wmcnamm=obj_get_metrics.WMCNAMM(classname)
                    wmcnamm = float(0 if wmcnamm is None else wmcnamm)
                    #if wmcnamm > 0:
                    WMCNAMM_List.append(wmcnamm)

                    # Calculate tcc
                    tcc=obj_get_metrics.TCC(classname)
                    tcc = float(0 if tcc is None else tcc)
                    #if tcc > 0:
                    TCC_List.append(tcc)

                    # Calculate atfd
                    atfd=obj_get_metrics.ATFD_CLASS(classname)
                    atfd = float(0 if atfd is None else atfd)
                    #if atfd > 0 :
                    ATFD_CLASS_List.append(atfd)

          # Calculate Method Metrics
          def methodmetrices():

              countmethod=len(db.ents("Method"))
              countm=1

              for methodname in db.ents("Method"):
                if (str(methodname.library()) != "Standard"):

                    #Print Execution State
                    exec = (str(countm) + "/" + str(countmethod))
                    countm += 1
                    sys.stdout.write('\r wait : ' + exec)

                    # Calculate loc
                    loc=obj_get_metrics.LOC(methodname)
                    loc = float(0 if loc is None else loc)
                    #if loc > 0:
                    LOC_List.append(loc)

                    # Calculate cyclo
                    cyclo=obj_get_metrics.CYCLO(methodname)
                    cyclo = float(0 if cyclo is None else cyclo)
                    #if cyclo > 0:
                    CYCLO_List.append(cyclo)

                    # Calculate maxnesting
                    maxnesting=obj_get_metrics.MAXNESTING(methodname)
                    maxnesting = float(0 if maxnesting is None else maxnesting)
                    #if maxnesting > 0:
                    MAXNESTING_List.append(maxnesting)

                    # Calculate nolv
                    nolv=obj_get_metrics.NOLV(methodname)
                    nolv = float(0 if nolv is None else nolv)
                    #if nolv > 0:
                    NOLV_List.append(nolv)

                    # Calculate atld
                    atld=obj_get_metrics.ATLD(db,methodname)
                    atld = float(0 if atld is None else atld)
                    #if atld > 0:
                    ATLD_List.append(atld)

                    # Calculate cc
                    cc=obj_get_metrics.give_cc(db,methodname)
                    cc = float(0 if cc is None else cc)
                    #if cc > 0:
                    CC_List.append(cc)

                    # Calculate cm
                    cm=obj_get_metrics.CM(methodname)
                    cm = float(0 if cm is None else cm)
                    #if cm > 0:
                    CM_List.append(cm)

                    # Calculate fanout
                    fanout=obj_get_metrics.FANOUT(methodname)
                    fanout = float(0 if fanout is None else fanout)
                    #if fanout > 0:
                    FANOUT_List.append(fanout)

                    # Calculate cint
                    cint=obj_get_metrics.CINT(methodname)
                    cint = float(0 if cint is None else cint)
                    #if cint > 0:
                    CINT_List.append(cint)

                    # Calculate cdisp
                    cdisp = obj_get_metrics.CDISP(methodname)
                    cdisp = float(0 if cdisp is None else cdisp)
                    #if cdisp > 0:
                    CDISP_List.append(cdisp)

                    # Calculate MaMCL
                    MaMCL = obj_get_metrics.MaMCL(methodname)
                    MaMCL = float(0 if MaMCL is None else MaMCL)
                    #if MaMCL > 0:
                    MaMCL_List.append(MaMCL)

                    # Calculate MeMCL
                    MeMCL = obj_get_metrics.MeMCL(methodname)
                    MeMCL = float(0 if MeMCL is None else MeMCL)
                    #if MeMCL > 0:
                    MeMCL_List.append(MeMCL)

                    # Calculate NMCS
                    NMCS = obj_get_metrics.CDISP(methodname)
                    NMCS = float(0 if NMCS is None else NMCS)
                    #if NMCS > 0:
                    NMCS_List.append(NMCS)

                    LAA = obj_get_metrics.LAA(methodname)
                    LAA = float(0 if LAA is None else LAA)
                    # if NMCS > 0:
                    LAA_List.append(LAA)

                    FDP = obj_get_metrics.FDP(methodname)
                    FDP = float(0 if FDP is None else FDP)
                    # if NMCS > 0:
                    FDP_List.append(FDP)

                    ATFDMETHOD = obj_get_metrics.ATFD_method(db,methodname)
                    ATFDMETHOD = float(0 if ATFDMETHOD is None else ATFDMETHOD)
                    # if NMCS > 0:
                    ATFDMETHOD_List.append(ATFDMETHOD)

          t1 = threading.Thread(target=classmetrices,name='t1')
          t2 = threading.Thread(target=methodmetrices,name='t2')
          t1.start()
          t2.start()
          t1.join()
          t2.join()


        #Save Results
        LOC = open('LOC.txt', 'w', newline='', encoding='utf-8')
        ATFD = open('ATFD.txt', 'w', newline='', encoding='utf-8')
        ATLD = open('ATLD.txt', 'w', newline='', encoding='utf-8')
        CC = open('CC.txt', 'w', newline='', encoding='utf-8')
        CM = open('CM.txt', 'w', newline='', encoding='utf-8')
        CINT = open('CINT.txt', 'w', newline='', encoding='utf-8')
        CDISP = open('CDISP.txt', 'w', newline='', encoding='utf-8')
        CYCLO = open('CYCLO.txt', 'w', newline='', encoding='utf-8')
        FANOUT = open('FANOUT.txt', 'w', newline='', encoding='utf-8')
        LOCNAMM = open('LOCNAMM.txt', 'w', newline='', encoding='utf-8')
        MaMCL = open('MaMCL.txt', 'w', newline='', encoding='utf-8')
        NMCS = open('NMCS.txt', 'w', newline='', encoding='utf-8')
        MAXNESTING = open('MAXNESTING.txt', 'w', newline='', encoding='utf-8')
        NOAM = open('NOAM.txt', 'w', newline='', encoding='utf-8')
        NOLV = open('NOLV.txt', 'w', newline='', encoding='utf-8')
        NOMNAMM = open('NOMNAMM.txt', 'w', newline='', encoding='utf-8')
        NOPA = open('NOPA.txt', 'w', newline='', encoding='utf-8')
        WOC = open('WOC.txt', 'w', newline='', encoding='utf-8')
        WMCNAMM = open('WMCNAMM.txt', 'w', newline='', encoding='utf-8')
        TCC = open('TCC.txt', 'w', newline='', encoding='utf-8')
        MeMCL = open('MeMCL.txt', 'w', newline='', encoding='utf-8')
        LAA = open('LAA.txt', 'w', newline='', encoding='utf-8')
        FDP = open('FDP.txt', 'w', newline='', encoding='utf-8')
        ATFDMETHOD = open('ATFDMETHOD.txt', 'w', newline='', encoding='utf-8')

        #Write Resut
        LOC.write(str(LOC_List))
        ATFD.write(str(ATFD_CLASS_List))
        ATLD.write(str(ATLD_List))
        CC.write(str(CC_List))
        CM.write(str(CM_List))
        CINT.write(str(CINT_List))
        CDISP.write(str(CDISP_List))
        CYCLO.write(str(CYCLO_List))
        FANOUT.write(str(FANOUT_List))
        LOCNAMM.write(str(LOCNAMM_List))
        MaMCL.write(str(MaMCL_List))
        MeMCL.write(str(MeMCL_List))
        NMCS.write(str(NMCS_List))
        MAXNESTING.write(str(MAXNESTING_List))
        NOAM.write(str(NOAM_List))
        NOLV.write(str(NOLV_List))
        NOMNAMM.write(str(NOMNAMM_List))
        NOPA.write(str(NOPA_List))
        WOC.write(str(WOC_List))
        WMCNAMM.write(str(WMCNAMM_List))
        TCC.write(str(TCC_List))
        LAA.write(str(LAA_List))
        FDP.write(str(FDP_List))
        ATFDMETHOD.write(str(ATFDMETHOD_List))

        print("\n End Process at " + str(datetime.now()))

    """Read Calculated Metrics From TXT Files"""
    def read_textFiles(self,metric):
        file = open(metric, "r+")
        temp = file.read()[1:-1]
        temp = temp.split(",")
        metrics_value = list()
        if not temp is None :
            for i in temp:
                metrics_value.append(float(i))
            return metrics_value



    """Save Thresholds"""
    def Save_Thresholds(self):
            print("Start Process at " + str(datetime.now()))
            metrics_name = ["LOCNAMM", "WMCNAMM", "NOMNAMM", "TCC", "WOC", "NOAM", "NOPA", "LOC", "CYCLO", "MAXNESTING", "NOLV", "ATLD", "CC", "CM", "FANOUT", "CINT", "CDISP", "MaMCL", "NMCS", "MeMCL", "ATFD", "LAA","FDP","ATFDMETHOD"]
            percentile=[.1,.25,.5,.75,.9]

            for metric in metrics_name:
                print('cals threshold for ' + metric)
                listresult = list()
                for percent in percentile:
                 qn = pd.Series(self.read_textFiles(metric+".txt")).quantile(percent)
                 listresult.append(qn)
                file = open(metric+"_Threshold.txt", 'w', newline='', encoding='utf-8')
                file.write(str(listresult))
            print("\n End Process at " + str(datetime.now()))


    """Save Udbs"""
    def create_understand_database_from_project(cls):
        #print("Start Process at " + str(datetime.now()))
        cmd = 'und create -db {0}{1}.udb -languages java add {2} analyze -all'
        root_path = os.getcwd()
        #print(os.getcwd())
        root_path = root_path +'\\'
        root_path = root_path.replace('\\','/')
        projects = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name)) if not os.path.isfile(os.path.join(root_path, name+".udb"))]
        for project in projects:
            #print("\n______________ project : ",project ,"________________")
            #print(cmd.format(root_path, project, root_path + project))
            command_ = cmd.format(root_path, project, root_path + project)
            os.system('cmd /c "{0}"'.format(command_))
        #print("\n End Process at " + str(datetime.now()))

    """Calc Codesmell """
    def Calc_Codesmell(self):

        #print("Start Process at " + str(datetime.now()))
        obj_get_metrics = All_Metrics.cls_get_metrics()

        ATFDTreshold = self.Get_Threshold("ATFD")
        ATLDTreshold = self.Get_Threshold("ATLD")
        CCTreshold = self.Get_Threshold("CC")
        CDISPTreshold = self.Get_Threshold("CDISP")
        CINTTreshold = self.Get_Threshold("CINT")
        CMTreshold = self.Get_Threshold("CM")
        CYCLOTreshold = self.Get_Threshold("CYCLO")
        NOLVTreshold = self.Get_Threshold("NOLV")
        FANOUTTreshold = self.Get_Threshold("FANOUT")
        LOCTreshold = self.Get_Threshold("LOC")
        LOCNAMMTreshold = self.Get_Threshold("LOCNAMM")
        MeMCLTreshold = self.Get_Threshold("MeMCL")
        MAXNESTINGTreshold = self.Get_Threshold("MAXNESTING")
        MaMCLTreshold = self.Get_Threshold("MaMCL")
        NMCSTreshold = self.Get_Threshold("NMCS")
        NOAMTreshold = self.Get_Threshold("NOAM")
        NOLVTreshold = self.Get_Threshold("NOLV")
        NOMNAMMTreshold = self.Get_Threshold("NOMNAMM")
        NOPATreshold = self.Get_Threshold("NOPA")
        TCCTreshold = self.Get_Threshold("TCC")
        WMCNAMMTreshold = self.Get_Threshold("WMCNAMM")
        WOCTreshold = self.Get_Threshold("WOC")
        LAATreshold = self.Get_Threshold("LAA")
        FDPTreshold = self.Get_Threshold("FDP")
        ATFDMETHODTreshold = self.Get_Threshold("ATFDMETHOD")

        def God_Class(input_class):
            listresult=list()

            f1=Metric_Condition(obj_get_metrics.LOCNAMM(input_class), LOCNAMMTreshold, "HIGH", "g")
            f2=Metric_Condition(obj_get_metrics.WMCNAMM(input_class),WMCNAMMTreshold,"MEAN","g")
            f3=Metric_Condition(obj_get_metrics.NOMNAMM(input_class),NOMNAMMTreshold,"HIGH","g")
            f4=Metric_Condition(obj_get_metrics.TCC(input_class), TCCTreshold, "LOW", "l")
            f5=Metric_Condition(obj_get_metrics.ATFD_CLASS(input_class), ATFDTreshold, "MEAN", "g")

            if(f1[0] and f2[0] and f3[0] and f4[0] and f5[0]):

                 intensity = ((f1[2] + f2[2] + f3[2] + f4[2] + f5[2]) / 5)
                 listresult.append(True)
                 listresult.append(intensity_label(intensity))
                 listresult.append(intensity)
                 listresult.append(f1[1] +f2[1]+f3[1]+f4[1]+f5[1])

            else:
                listresult.append(False)
                listresult.append(0)
                listresult.append(0)
                listresult.append(0)

            return listresult

        def Data_Class(input_class):
            listresult = list()

            f1=Metric_Condition(obj_get_metrics.WMCNAMM(input_class), WMCNAMMTreshold, "LOW", "l")
            f2=Metric_Condition(obj_get_metrics.WOC(input_class), WOCTreshold, "LOW", "l")
            f3=Metric_Condition(obj_get_metrics.NOAM(input_class), NOAMTreshold, "MEAN", "g")
            f4=Metric_Condition(obj_get_metrics.NOPA(input_class), NOPATreshold, "MEAN", "g")

            if (f1[0] and f2[0] and f3[0] and f4[0]):
                intensity=((f1[2] + f2[2] + f3[2] + f4[2]) / 4)
                listresult.append(True)
                listresult.append(intensity_label(intensity))
                listresult.append(intensity)
                listresult.append(f1[1] + f2[1] + f3[1] + f4[1] )

            else:
                listresult.append(False)
                listresult.append(0)
                listresult.append(0)
                listresult.append(0)

            return listresult



        def Brain_Method(input_Method):
            listresult = list()

            f1=Metric_Condition(obj_get_metrics.LOC(input_Method), LOCTreshold, "HIGH", "g")
            f2=Metric_Condition(obj_get_metrics.CYCLO(input_Method), CYCLOTreshold, "HIGH", "g")
            f3=Metric_Condition(obj_get_metrics.MAXNESTING(input_Method), MAXNESTINGTreshold, "HIGH", "g")
            f4=Metric_Condition(obj_get_metrics.NOLV(input_Method), NOLVTreshold, "MEAN", "g")
            f5=Metric_Condition(obj_get_metrics.ATLD(db,input_Method), ATLDTreshold, "MEAN", "g")

            if ((f1[0] and f2[0] and f3[0]) or (f4[0] and f5[0])):
                intensity = ((f1[2] + f2[2] + f3[2] + f4[2] + f5[2]) / 5)
                listresult.append(True)
                listresult.append(intensity_label(intensity))
                listresult.append(intensity)
                listresult.append(f1[1] + f2[1] + f3[1] + f4[1] + f5[1])

            else:
                listresult.append(False)
                listresult.append(0)
                listresult.append(0)
                listresult.append(0)

            return listresult

        def Shotgun_Surgery(input_Method):

            listresult = list()
            f1=Metric_Condition(obj_get_metrics.give_cc(db,input_Method), CCTreshold, "HIGH", "g")
            f2=Metric_Condition(obj_get_metrics.CM(input_Method), CMTreshold, "HIGH", "g")
            f3=Metric_Condition(obj_get_metrics.FANOUT(input_Method), FANOUTTreshold, "LOW", "g")

            if (f1[0] and f2[0] and f3[0]):
                intensity = ((f1[2] + f2[2] + f3[2] ) / 3)
                listresult.append(True)
                listresult.append(intensity_label(intensity))
                listresult.append(intensity)
                listresult.append(f1[1] + f2[1] + f3[1] )

            else:
                listresult.append(False)
                listresult.append(0)
                listresult.append(0)

            return listresult

        def Dispersed_Coupling(input_Method):
            listresult = list()

            f1=Metric_Condition(obj_get_metrics.CINT(input_Method), CINTTreshold, "HIGH", "g")
            f2=Metric_Condition(obj_get_metrics.CDISP(input_Method), CDISPTreshold, "HIGH", "g")

            if (f1[0] and f2[0]):
                intensity=((f1[2] + f2[2] ) / 2)
                listresult.append(True)
                listresult.append(intensity_label(intensity))
                listresult.append(intensity)
                listresult.append(f1[1] + f2[1] )

            else:
                listresult.append(False)
                listresult.append(0)
                listresult.append(0)
                listresult.append(0)

            return listresult

        def Message_Chains(input_Method):
            listresult = list()

            f1=Metric_Condition(obj_get_metrics.MaMCL(input_Method), MaMCLTreshold, "MEAN", "g")
            f2=Metric_Condition(obj_get_metrics.NMCS(input_Method), NMCSTreshold, "MEAN", "g")
            f3=Metric_Condition(obj_get_metrics.MeMCL(input_Method), MeMCLTreshold, "LOW", "g")
            if (f1[0] or (f2[0] and f3[0] ) ):
                intensity = ((f1[2] + f2[2] + f3[2]) / 3)
                listresult.append(True)
                listresult.append(intensity_label(intensity))
                listresult.append(intensity)
                listresult.append(f1[1] + f2[1] + f3[1])

            else:
                listresult.append(False)
                listresult.append(0)
                listresult.append(0)
                listresult.append(0)

            return listresult


        def Feature_Envy(input_Method):
            listresult = list()

            f1=Metric_Condition(obj_get_metrics.ATFD_method(db,input_Method), ATFDMETHODTreshold, "MEAN", "g")
            f2=Metric_Condition(obj_get_metrics.LAA(input_Method), LAATreshold, "LOW", "l")
            f3=Metric_Condition(obj_get_metrics.FDP(input_Method), FDPTreshold, "MEAN", "l")
            if (f1[0] or (f2[0] and f3[0] ) ):
                intensity = ((f1[2] + f2[2] + f3[2]) / 3)
                listresult.append(True)
                listresult.append(intensity_label(intensity))
                listresult.append(intensity)
                listresult.append(f1[1] + f2[1] + f3[1])

            else:
                listresult.append(False)
                listresult.append(0)
                listresult.append(0)
                listresult.append(0)

            return listresult



        """Check Metric Condition"""
        def Metric_Condition(Metric, Threshold, Kind, Operator):

            map = {"VERYLOW": "0", "LOW": "1", "MEAN": "2", "HIGH": "3", "VERYHIGHT": "4"}
            mapkind = int(map[Kind])
            metric = float(0 if Metric is None else Metric)
            threshold = Threshold[mapkind]

            listresult = list()
            status=False
            ER=1
            inensity = 0

            #Calc Status
            if (Operator == "g"):
                if (metric >= threshold):
                    status = True
            elif (Operator == "l"):
                if (metric <= threshold):
                    status = True

            if status :

             #Calc ER
             if(threshold>0):
               ER = math.ceil(metric/threshold)
               if ER == 0 :
                    ER=1

             #calc intensity
             x = range(5)
             for n in x :
              if(metric >= Threshold[n]):
                 inensity= n

             inensity=return_intensity(inensity)

            listresult.append(status)
            listresult.append(ER)
            listresult.append(inensity)

            return listresult

        #Intensity Label
        def intensity_label(intensity):
            if intensity >=1 and intensity < 3.25:
                return "Very Low"

            elif intensity >= 3.25 and intensity < 5.5:
                return "Low"

            elif intensity >= 5.5 and intensity < 7.75:
                return "MEAN"

            elif intensity >= 7.75 and intensity < 10 :
                return "HIGH"

            elif intensity ==10:
                return "Very HIGH"

        # Intensity Label
        def return_intensity(intensity):
            if intensity == 0:
                return 1

            elif intensity ==1:
                return 3.25

            elif intensity ==2:
                return 5.5

            elif intensity ==3:
                return 7.75

            elif intensity == 4:
                return 10

        #Define List To Save Results
        God_Classlist=list()
        Data_Classlist=list()
        Brain_Methodlist=list()
        Shotgun_Surgerylist=list()
        Dispersed_Couplinglist=list()
        Message_Chainslist=list()
        Feature_Envylist = list()

        #setHeaders
        God_Classlist.append(["ProjectName","ClassName","IntensityLabel","Intensity","ER"])
        Data_Classlist.append(["ProjectName", "ClassName", "IntensityLabel", "Intensity", "ER"])
        Brain_Methodlist.append(["ProjectName", "ClassName", "IntensityLabel", "Intensity", "ER"])
        Shotgun_Surgerylist.append(["ProjectName", "ClassName", "IntensityLabel", "Intensity", "ER"])
        Dispersed_Couplinglist.append(["ProjectName", "ClassName", "IntensityLabel", "Intensity", "ER"])
        Message_Chainslist.append(["ProjectName", "ClassName", "IntensityLabel", "Intensity", "ER"])
        Feature_Envylist.append(["ProjectName", "ClassName", "IntensityLabel", "Intensity", "ER"])

        """Start Open Udb Files And Calculate Class CodeSmells"""
        file_list = list()
        for file in glob.glob("*.udb"):
            file_list.append(file)

        """ start Calculate Projects Code Smells """
        for project in file_list:
          db = understand.open(project)

          #print("\n______________ project : ", project, "________________")

          # Calculate Class Codesmells
          def Calc_ClassCodesmells():
            countclass = len(db.ents("class"))
            count = 1

            for inputclass in db.ents("class"):

                # Print Execution State
                exec = (str(count) + "/" + str(countclass))
                count += 1
                sys.stdout.write('\r wait : ' + exec)


                #Calcualte God Class
                if (str(inputclass.library()) != "Standard" and God_Class(inputclass)[0]):
                    resultlist = list()
                    resultlist.append(project)
                    resultlist.append(inputclass.longname())
                    resultlist.append(str(God_Class(inputclass)[1]))
                    resultlist.append(str(God_Class(inputclass)[2]))
                    resultlist.append(str(God_Class(inputclass)[3]))

                    God_Classlist.append(resultlist)

                #Calcualte Data Class
                if (str(inputclass.library()) != "Standard" and Data_Class(inputclass)[0]):
                    resultlist = list()
                    resultlist.append(project)
                    resultlist.append(inputclass.longname())
                    resultlist.append(str(Data_Class(inputclass)[1]))
                    resultlist.append(str(Data_Class(inputclass)[2]))
                    resultlist.append(str(Data_Class(inputclass)[3]))

                    Data_Classlist.append(resultlist)
          #print('aaaa')
          # Calculate Method Codesmells
          def Calc_methodCodesmells():
            countmethod = len(db.ents("Method"))
            count = 1

            for inputmethod in db.ents("Method"):

                # Print Execution State
                exec = (str(count) + "/" + str(countmethod))
                count += 1
                sys.stdout.write('\r wait : ' + exec)

                #Calcualte Brain Method
                if (str(inputmethod.library()) != "Standard" and Brain_Method(inputmethod)[0]):
                    resultlist = list()
                    resultlist.append(project)
                    resultlist.append(inputmethod.longname())
                    resultlist.append(str(Brain_Method(inputmethod)[1]))
                    resultlist.append(str(Brain_Method(inputmethod)[2]))
                    resultlist.append(str(Brain_Method(inputmethod)[3]))

                    Brain_Methodlist.append(resultlist)

                #Calcualte Shotgun Surgery
                if (str(inputmethod.library()) != "Standard" and Shotgun_Surgery(inputmethod)[0]):
                    resultlist = list()
                    resultlist.append(project)
                    resultlist.append(inputmethod.longname())
                    resultlist.append(str(Shotgun_Surgery(inputmethod)[1]))
                    resultlist.append(str(Shotgun_Surgery(inputmethod)[2]))
                    resultlist.append(str(Shotgun_Surgery(inputmethod)[3]))

                    Shotgun_Surgerylist.append(resultlist)

                #Calcualte Dispersed Coupling
                if (str(inputmethod.library()) != "Standard" and Dispersed_Coupling(inputmethod)[0]):
                    resultlist = list()
                    resultlist.append(project)
                    resultlist.append(inputmethod.longname())
                    resultlist.append(str(Dispersed_Coupling(inputmethod)[1]))
                    resultlist.append(str(Dispersed_Coupling(inputmethod)[2]))
                    resultlist.append(str(Dispersed_Coupling(inputmethod)[3]))

                    Dispersed_Couplinglist.append(resultlist)

                #Calcualte Message Chains
                if (str(inputmethod.library()) != "Standard" and Message_Chains(inputmethod)[0]):
                    resultlist = list()
                    resultlist.append(project)
                    resultlist.append(inputmethod.longname())
                    resultlist.append(str(Message_Chains(inputmethod)[1]))
                    resultlist.append(str(Message_Chains(inputmethod)[2]))
                    resultlist.append(str(Message_Chains(inputmethod)[3]))

                    Message_Chainslist.append(resultlist)

                if (str(inputmethod.library()) != "Standard" and Feature_Envy(inputmethod)[0]):
                    resultlist = list()
                    resultlist.append(project)
                    resultlist.append(inputmethod.longname())
                    resultlist.append(str(Feature_Envy(inputmethod)[1]))
                    resultlist.append(str(Feature_Envy(inputmethod)[2]))
                    resultlist.append(str(Feature_Envy(inputmethod)[3]))

                    Feature_Envylist.append(resultlist)

          t1 = threading.Thread(target=Calc_ClassCodesmells, name='t1')
          t2 = threading.Thread(target=Calc_methodCodesmells, name='t2')
          t1.start()
          t2.start()
          t1.join()
          t2.join()

        #Save God Class To csv File
        with open('GodClass_Codesmells.csv', 'w') as file:
           for line in God_Classlist:
              file.write(",".join(line))
              file.write('\n')

        #Save Data Class To csv File
        with open('DataClass_Codesmells.csv', 'w') as file:
           for line in Data_Classlist:
              file.write(",".join(line))
              file.write('\n')

        #Save Brain Method To csv File
        with open('BrainMethod_Codesmells.csv', 'w') as file:
           for line in Brain_Methodlist:
              file.write(",".join(line))
              file.write('\n')

        #Save Shutgun Suegery To csv File
        with open('ShutgunSurgery_Codesmells.csv', 'w') as file:
           for line in Shotgun_Surgerylist:
              file.write(",".join(line))
              file.write('\n')

        #Save Dispersed Coupling To csv File
        with open('DispersedCoupling_Codesmells.csv', 'w') as file:
           for line in Dispersed_Couplinglist:
              file.write(",".join(line))
              file.write('\n')

        #Save Feature Envy To csv File
        with open('FeatureEnvy_Codesmells.csv', 'w') as file:
           for line in Feature_Envylist:
              file.write(",".join(line))
              file.write('\n')


        #print("\n End Process at " + str(datetime.now()))



    """Get Threshold From TXT Files"""
    def Get_Threshold(self,Metric):
     return  self.read_textFiles(Metric + "_Threshold.txt")

    def extract_metrics_and_coverage_all(cls, udbs_path=str,
                                         class_list_csv_path=None,
                                         csvs_path=str,
                                         ):
        df = class_list_csv_path
        files = [f for f in os.listdir(udbs_path) if os.path.isfile(os.path.join(udbs_path, f))]
        t = list()
        p = list()
        for i, f in enumerate(files):
         if f.endswith(".udb") and not os.path.isfile(os.path.join(udbs_path, os.path.splitext(f)[0]+'.csv')) :
            print('processing understand db file {0}:'.format(f))
            #print('here')
            db = understand.open(os.path.join(udbs_path, f))
            #print(db)
            #df = db.ents('Java Class ~Interface ~Enum ~Unknown ~Unresolved ~Jar ~Library')
            df = db.ents('Java Class ~Enum ~Unknown ~Unresolved ~Jar ~Library')
            #print(df)
            cls.compute_metrics_by_class_list(project_name=f[:-4], database=db, class_list=df, csv_path=csvs_path)
            print('processing understand db file {0} was finished'.format(f))

    def compute_metrics_by_class_list(cls, project_name: str = None, database=None, class_list=None, csv_path=None):
        all_class_metrics_value = list()

        for class_entity in class_list:
          print("\n Start Class at " +class_entity.longname() + str(datetime.now()))
          print()
          #print ('Before method_list')
          method_list = UnderstandUtility.get_method_of_class_java2(db=database, class_name=class_entity.longname(),
                                                                      class_entity=class_entity)
          #print('After method_list')
          #print(class_entity.longname())
          #print(class_entity)
          #print(method_list)
          #print('Class Name : '+class_entity.longname())
          #print('Method Count : '+ str(len(method_list)))

          #if len(method_list)>0:
            #print('Calculating Project metrics')
            #project_metrics_dict = cls.compute_java_project_metrics(db=database)
            #if project_metrics_dict is None:
            #    raise TypeError('No project metric for item {} was found'.format(class_entity.longname()))

            #nt('Calculating package metrics')
            #package_metrics_dict = cls.compute_java_package_metrics(db=database,
            #                                                                       class_name=class_entity.longname())
            #if package_metrics_dict is None:
            #    raise TypeError('No package metric for item {} was found'.format(class_entity.longname()))

            #print('Calculating class lexicon metrics')
            #class_lexicon_metrics_dict = cls.compute_java_class_metrics_lexicon(db=database,
             #                                                                                  entity=class_entity)
            #if class_lexicon_metrics_dict is None:
             #   raise TypeError('No class lexicon metric for item {} was found'.format(class_entity.longname()))

            #print('Calculating class ordinary')
            #print(class_entity)
            #print(database)
            #print('compute_java_class_metrics2')
            #class_ordinary_metrics_dict = cls.compute_java_class_metrics2(db=database,
            #                                                                             entity=class_entity)



           # if class_ordinary_metrics_dict is None:
            #    raise TypeError('No class ordinary metric for item {} was found'.format(class_entity.longname()))


            #print(class_entity)
            #print(database)
            #print(class_entity.longname())
          #print('Calculating class get_method_of_class_java2')
          #method_list = UnderstandUtility.get_method_of_class_java2(db=database,class_name=class_entity.longname(),class_entity=class_entity)

           #print('Calculating class get_method_of_class_java2')

          for method in method_list:
            #if method.longname() == "com.alibaba.druid.sql.ast.statement.SQLUpdateStatement.setTableSource":
             print("\n Start method at " + method.longname() + str(datetime.now()))
             #print(type(method))
             #print(method.kind())
             #return
             print('Before method_ordinary_metrics_dict'+ str(datetime.now()))
             method_ordinary_metrics_dict = cls.compute_java_method_metrics(db=database,
                                                                                         entity=method)
             print('After method_ordinary_metrics_dict'+ str(datetime.now()))
             #method_Lexicon_metrics_dict = cls.compute_java_class_metrics_lexicon(db=database,
                                                                            #entity=method)
             print('Before one_class_metrics_value'+ str(datetime.now()))
             one_class_metrics_value = [UnderstandUtility.get_project_files_java(database)[0]
                 , UnderstandUtility.get_package_of_given_class(db=database,class_name=class_entity.longname()),
                                        class_entity.longname(),method.simplename()]
             print('After one_class_metrics_value'+ str(datetime.now()))
             #print('Write package_metrics_dict')
             #for metric_name in cls.get_project_metrics_names():
              #   one_class_metrics_value.append(project_metrics_dict[metric_name])

             # Write package_metrics_dict
             #for metric_name in cls.get_package_metrics_names():
              #  one_class_metrics_value.append(package_metrics_dict[metric_name])

             # Write class_lexicon_metrics_dict
             #for metric_name in cls.get_class_lexicon_metrics_names():
              #  one_class_metrics_value.append(class_lexicon_metrics_dict[metric_name])

             # Write class_ordinary_metrics_dict
             #for metric_name in cls.get_class_ordinary_metrics_names():
              #  one_class_metrics_value.append(class_ordinary_metrics_dict[metric_name])

             # Write method_ordinary_metrics_dict
             for metric_name in cls.get_method_metrics_names():
                one_class_metrics_value.append(method_ordinary_metrics_dict[metric_name])

             # Write method_Lexicon_metrics_dict
             #for metric_name in cls.get_class_lexicon_metrics_names():
                 #one_class_metrics_value.append(method_Lexicon_metrics_dict[metric_name])

             all_class_metrics_value.append(one_class_metrics_value)

        #print('Calculating class columns')

        columns = ['Project','Package','Class','Method']
        columns.extend(cls.get_all_metrics_names())
        df = pd.DataFrame(data=all_class_metrics_value, columns=columns)
        #print('df for class {0} with shape {1}'.format(project_name, df.shape))
        df.to_csv(csv_path + project_name + '.csv', index=False)



    def get_method_metrics_names(cls) -> list:
        return metrics_names.method_metrics_names


    def get_class_ordinary_metrics_names(cls) -> list:
        return metrics_names.class_ordinary_metrics_names


    def get_class_lexicon_metrics_names(cls) -> list:
        return metrics_names.class_lexicon_metrics_names


    def get_package_metrics_names(cls) -> list:
        return metrics_names.package_metrics_names


    def get_project_metrics_names(cls) -> list:
        return metrics_names.project_metrics_names



    def get_all_metrics_names(cls) -> list:
        metrics = list()

        #for metric_name in cls.get_project_metrics_names():
         #   metrics.append('PJ_' + metric_name)

        #for metric_name in cls.get_package_metrics_names():
         #   metrics.append('PK_' + metric_name)

        #for metric_name in cls.get_class_lexicon_metrics_names():
         #   metrics.append('CSLEX_' + metric_name)

        #for metric_name in cls.get_class_ordinary_metrics_names():
         #   metrics.append('CSORD_' + metric_name)

        for metric_name in cls.get_method_metrics_names():
            metrics.append('MD_' + metric_name)

        #for metric_name in cls.get_class_lexicon_metrics_names():
            #metrics.append('MDLEX_' + metric_name)

        return metrics



    def compute_java_method_metrics(cls, db=None, entity=None):

        methodmetrics = dict()


        j_code_odor_metric = JCodeOdorMetric()
        print('Before ATFD' + str(datetime.now()))
        methodmetrics.update({'ATFD': j_code_odor_metric.ATFD_method(db,entity)})
        print('Before FDP' + str(datetime.now()))
        methodmetrics.update({'FDP': j_code_odor_metric.fdp(entity)})
        print('Before CFNAMM' + str(datetime.now()))
        methodmetrics.update({'CFNAMM': j_code_odor_metric.cfnamm_method(entity)})
        print('Before ATLD' + str(datetime.now()))
        methodmetrics.update({'ATLD': j_code_odor_metric.ATLD(db,entity)})
        print('Before CC' + str(datetime.now()))
        methodmetrics.update({'CC': j_code_odor_metric.give_cc(db,entity)})
        print('Before CDISP' + str(datetime.now()))
        methodmetrics.update({'CDISP': j_code_odor_metric.CDISP(entity)})
        print('Before CINT' + str(datetime.now()))
        methodmetrics.update({'CINT': j_code_odor_metric.CINT(entity)})
        print('Before CM' + str(datetime.now()))
        methodmetrics.update({'CM': j_code_odor_metric.CM(entity)})
        print('Before CYCLO' + str(datetime.now()))
        methodmetrics.update({'CYCLO': j_code_odor_metric.CYCLO(entity)})
        print('Before FANIN' + str(datetime.now()))
        methodmetrics.update({'FANIN': j_code_odor_metric.FANIN_Method(entity)})
        print('Before FANOUT' + str(datetime.now()))
        methodmetrics.update({'FANOUT': j_code_odor_metric.FANOUT_Method(entity)})
        print('Before LOC' + str(datetime.now()))
        methodmetrics.update({'LOC': j_code_odor_metric.LOC(entity)})
        print('Before MaMCL' + str(datetime.now()))
        methodmetrics.update({'MaMCL': j_code_odor_metric.MaMCL(entity)})
        print('Before MeMCL' + str(datetime.now()))
        methodmetrics.update({'MeMCL': j_code_odor_metric.Memcl(entity)})
        print('Before NMCS' + str(datetime.now()))
        methodmetrics.update({'NMCS': j_code_odor_metric.NMCS(entity)})
        print('Before NOLV' + str(datetime.now()))
        methodmetrics.update({'NOLV': j_code_odor_metric.NOLV(entity)})
        print('Before LOCNAMM' + str(datetime.now()))
        methodmetrics.update({'LOCNAMM': j_code_odor_metric.LOCNAMM(entity)}) #edit
        print('Before LAA' + str(datetime.now()))
        methodmetrics.update({'LAA': j_code_odor_metric.laa(entity)})
        print('Before MAXNESTING' + str(datetime.now()))
        methodmetrics.update({'MAXNESTING': j_code_odor_metric.MAXNESTING(entity)})
        print('Before CLNAMM' + str(datetime.now()))
        methodmetrics.update({'CLNAMM': j_code_odor_metric.clnamm(entity)})
        print('Before NOP' + str(datetime.now()))
        methodmetrics.update({'NOP': j_code_odor_metric.nop(entity)})
        print('Before NOAV' + str(datetime.now()))
        methodmetrics.update({'NOAV': j_code_odor_metric.noav(entity)})
        print('Before CountLineBlank' + str(datetime.now()))
        methodmetrics.update({'CountLineBlank': j_code_odor_metric.CountLineBlank(entity)})
        print('Before NL' + str(datetime.now()))
        methodmetrics.update({'NL': j_code_odor_metric.NL(entity)})
        print('Before CountLineCodeDecl' + str(datetime.now()))
        methodmetrics.update({'CountLineCodeDecl': j_code_odor_metric.CountLineCodeDecl(entity)})
        print('Before CountLineCodeExe' + str(datetime.now()))
        methodmetrics.update({'CountLineCodeExe': j_code_odor_metric.CountLineCodeExe(entity)})
        print('Before NPATH' + str(datetime.now()))
        methodmetrics.update({'NPATH': j_code_odor_metric.NPATH(entity)})
        print('Before CountStmt' + str(datetime.now()))
        methodmetrics.update({'CountStmt': j_code_odor_metric.CountStmt(entity)})
        print('Before CountStmtDecl' + str(datetime.now()))
        methodmetrics.update({'CountStmtDecl': j_code_odor_metric.CountStmtDecl(entity)})
        print('Before CountStmtExe' + str(datetime.now()))
        methodmetrics.update({'CountStmtExe': j_code_odor_metric.CountStmtExe(entity)})
        print('Before CyclomaticStrict' + str(datetime.now()))
        methodmetrics.update({'CyclomaticStrict': j_code_odor_metric.CyclomaticStrict(entity)})
        print('Before CyclomaticModified' + str(datetime.now()))
        methodmetrics.update({'CyclomaticModified': j_code_odor_metric.CyclomaticModified(entity)}) #edit
        print('Before Essential' + str(datetime.now()))
        methodmetrics.update({'Essential': j_code_odor_metric.Essential(entity)})
        print('Before Knots' + str(datetime.now()))
        methodmetrics.update({'Knots': j_code_odor_metric.Knots(entity)})
        print('Before MaxEssentialKnots' + str(datetime.now()))
        methodmetrics.update({'MaxEssentialKnots': j_code_odor_metric.MaxEssentialKnots(entity)})
        print('Before RatioCommentToCode' + str(datetime.now()))
        methodmetrics.update({'RatioCommentToCode': j_code_odor_metric.RatioCommentToCode(entity)})
        print('Before CountPathLog' + str(datetime.now()))
        methodmetrics.update({'CountPathLog': j_code_odor_metric.CountPathLog(entity)})

        cls.remove_none_from_lists([methodmetrics])
        return methodmetrics

        #return class_metrics


    def compute_java_class_metrics2(cls, db=None, entity=None):

        # 1. Understand built-in class metrics
        #print(entity)
        #print(entity.metrics())
        #print(entity.metric(entity.metrics()))
        class_metrics = entity.metric(entity.metrics())
        #print('heeee')
        # 2. Systematically created metrics
        j_code_odor_metric = JCodeOdorMetric()
        #print(entity.longname())
        method_list = UnderstandUtility.get_method_of_class_java2(db=db, class_name=entity.longname(),class_entity=entity)
        #print(method_list)
        if method_list is None:
            raise TypeError('method_list is none for class "{}"'.format(entity.longname()))

        # 2.1 CSCC
        class_cyclomatic_list = list()
        class_cyclomatic_namm_list = list()

        class_cyclomatic_strict_list = list()
        class_cyclomatic_strict_namm_list = list()

        class_cyclomatic_modified_list = list()
        class_cyclomatic_modified_namm_list = list()

        class_essential_list = list()
        class_essential_namm_list = list()
        #print(method_list)
        for method in method_list:
            class_cyclomatic_list.append(method.metric(['Cyclomatic'])['Cyclomatic'])
            class_cyclomatic_strict_list.append(method.metric(['CyclomaticStrict'])['CyclomaticStrict'])
            class_cyclomatic_modified_list.append(method.metric(['CyclomaticModified'])['CyclomaticModified'])
            class_essential_list.append(method.metric(['Essential'])['Essential'])
            if not j_code_odor_metric.is_accesor_or_mutator(input_method=method):
                class_cyclomatic_namm_list.append(method.metric(['Cyclomatic'])['Cyclomatic'])
                class_cyclomatic_strict_namm_list.append(method.metric(['CyclomaticStrict'])['CyclomaticStrict'])
                class_cyclomatic_modified_namm_list.append(method.metric(['CyclomaticModified'])['CyclomaticModified'])
                class_essential_namm_list.append(method.metric(['Essential'])['Essential'])

        cls.remove_none_from_lists([class_cyclomatic_list, class_cyclomatic_namm_list,
                                    class_cyclomatic_strict_list, class_cyclomatic_strict_namm_list,
                                    class_cyclomatic_modified_list, class_cyclomatic_modified_namm_list,
                                    class_essential_list, class_essential_namm_list])

        # CSCC
        # 2.1.13
        class_metrics.update({'MinCyclomatic': min(class_cyclomatic_list)})
        # 2.1.14
        class_metrics.update({'MinCyclomaticStrict': min(class_cyclomatic_strict_list)})
        # 2.1.15
        class_metrics.update({'MinCyclomaticModified': min(class_cyclomatic_modified_list)})
        # 2.1.16
        class_metrics.update({'MinEssential': min(class_essential_list)})

        # 2.1.17
        class_metrics.update({'SDCyclomatic': np.std(class_cyclomatic_list)})
        # 2.1.18
        class_metrics.update({'SDCyclomaticStrict': np.std(class_cyclomatic_strict_list)})
        # 2.1.19
        class_metrics.update({'SDCyclomaticModified': np.std(class_cyclomatic_modified_list)})
        # 2.1.20
        class_metrics.update({'SDEssential': np.std(class_essential_list)})

        class_metrics.update({'LogCyclomatic': math.log10(sum(class_cyclomatic_list)+1)})
        class_metrics.update({'LogCyclomaticStrict': math.log10(sum(class_cyclomatic_strict_list)+1)})
        class_metrics.update({'LogCyclomaticModified': math.log10(sum(class_cyclomatic_modified_list)+1)})
        class_metrics.update({'LogEssential': math.log10(sum(class_essential_list)+1)})

        # CSCCNAMM
        # 2.1.21
        class_metrics.update({'SumCyclomaticNAMM': sum(class_cyclomatic_namm_list)})
        # 2.1.22
        class_metrics.update({'SumCyclomaticStrictNAMM': sum(class_cyclomatic_strict_namm_list)})
        # 2.1.23
        class_metrics.update({'SumCyclomaticModifiedNAMM': sum(class_cyclomatic_modified_namm_list)})
        # 2.1.24
        class_metrics.update({'SumEssentialNAMM': sum(class_essential_namm_list)})

        # 2.1.25
        class_metrics.update({'MaxCyclomaticNAMM': max(class_cyclomatic_namm_list)})
        # 2.1.26
        class_metrics.update({'MaxCyclomaticStrictNAMM': max(class_cyclomatic_strict_namm_list)})
        # 2.1.27
        class_metrics.update({'MaxCyclomaticModifiedNAMM': max(class_cyclomatic_modified_namm_list)})
        # 2.1.28
        class_metrics.update({'MaxEssentialNAMM': max(class_essential_namm_list)})

        # 2.1.29
        class_metrics.update({'AvgCyclomaticNAMM': sum(class_cyclomatic_namm_list) / len(class_cyclomatic_namm_list)})
        # 2.1.30
        class_metrics.update({'AvgCyclomaticStrictNAMM': sum(class_cyclomatic_strict_namm_list) / len(
            class_cyclomatic_strict_namm_list)})
        # 2.1.31
        class_metrics.update({'AvgCyclomaticModifiedNAMM': sum(class_cyclomatic_modified_namm_list) / len(
            class_cyclomatic_modified_namm_list)})
        # 2.1.32
        class_metrics.update({'AvgEssentialNAMM': sum(class_essential_namm_list) / len(class_essential_namm_list)})

        # 2.1.33
        class_metrics.update({'MinCyclomaticNAMM': min(class_cyclomatic_namm_list)})
        # 2.1.34
        class_metrics.update({'MinCyclomaticStrictNAMM': min(class_cyclomatic_strict_namm_list)})
        # 2.1.35
        class_metrics.update({'MinCyclomaticModifiedNAMM': min(class_cyclomatic_modified_namm_list)})
        # 2.1.36
        class_metrics.update({'MinEssentialNAMM': min(class_essential_namm_list)})

        # 2.1.37
        class_metrics.update({'SDCyclomaticNAMM': np.std(class_cyclomatic_namm_list)})
        # 2.1.38
        class_metrics.update({'SDCyclomaticStrictNAMM': np.std(class_cyclomatic_strict_namm_list)})
        # 2.1.39
        class_metrics.update({'SDCyclomaticModifiedNAMM': np.std(class_cyclomatic_modified_namm_list)})
        # 2.1.40
        class_metrics.update({'SDEssentialNAMM': np.std(class_essential_namm_list)})

        # 2.2 CSNOP (10)
        #
        parameters_length_list = list()
        parameters_length_namm_list = list()
        # number_of_parameters = 0
        # print('method list', len(method_list))
        for method in method_list:
            # if method.library() != "Standard":
            # print('method params', method.longname(), '-->', method.parameters())
            params = method.parameters().split(',')
            if len(params) == 1:
                if params[0] == ' ' or params[0] == '' or params[0] is None:
                    parameters_length_list.append(0)
                else:
                    parameters_length_list.append(1)
            else:
                parameters_length_list.append(len(params))

            if not j_code_odor_metric.is_accesor_or_mutator(input_method=method):
                if len(params) == 1:
                    if params[0] == ' ' or params[0] == '' or params[0] is None:
                        parameters_length_namm_list.append(0)
                    else:
                        parameters_length_namm_list.append(1)
                else:
                    parameters_length_namm_list.append(len(params))

        cls.remove_none_from_lists([parameters_length_list, parameters_length_namm_list])

        # print('number of parameters', number_of_parameters)
        # CSNOP
        # 2.2.1
        class_metrics.update({'SumCSNOP': sum(parameters_length_list)})
        # 2.2.2
        class_metrics.update({'MaxCSNOP': max(parameters_length_list)})
        # 2.2.3
        class_metrics.update({'MinCSNOP': min(parameters_length_list)})
        # 2.2.4
        class_metrics.update({'AvgCSNOP': sum(parameters_length_list) / len(parameters_length_list)})
        # 2.2.5
        class_metrics.update({'SDCSNOP': np.std(parameters_length_list)})

        # CSNOP_NAMM
        # 2.2.6
        class_metrics.update({'SumCSNOPNAMM': sum(parameters_length_namm_list)})
        # 2.2.7
        class_metrics.update({'MaxCSNOPNAMM': max(parameters_length_namm_list)})
        # 2.2.8
        class_metrics.update({'MinCSNOPNAMM': min(parameters_length_namm_list)})
        # 2.2.9
        class_metrics.update({'AvgCSNOPNAMM': sum(parameters_length_namm_list) / len(parameters_length_namm_list)})
        # 2.2.10
        class_metrics.update({'SDCSNOPNAMM': np.std(parameters_length_namm_list)})

        # 2.3 SCLOC (30)
        #
        line_of_code_list = list()
        line_of_code_namm_list = list()

        line_of_code_decl_list = list()
        line_of_code_decl_namm_list = list()

        line_of_code_exe_list = list()
        line_of_code_exe_namm_list = list()
        for method in method_list:
            line_of_code_list.append(method.metric(['CountLineCode'])['CountLineCode'])
            line_of_code_decl_list.append(method.metric(['CountLineCodeDecl'])['CountLineCodeDecl'])
            line_of_code_exe_list.append(method.metric(['CountLineCodeExe'])['CountLineCodeExe'])
            if not j_code_odor_metric.is_accesor_or_mutator(input_method=method):
                line_of_code_namm_list.append(method.metric(['CountLineCode'])['CountLineCode'])
                line_of_code_decl_namm_list.append(method.metric(['CountLineCodeDecl'])['CountLineCodeDecl'])
                line_of_code_exe_namm_list.append(method.metric(['CountLineCodeExe'])['CountLineCodeExe'])

        cls.remove_none_from_lists([line_of_code_list, line_of_code_namm_list,
                                    line_of_code_decl_list, line_of_code_decl_namm_list,
                                    line_of_code_exe_list, line_of_code_exe_namm_list])
        # CSLOC_All
        # 2.3.5
        class_metrics.update({'AvgLineCodeDecl': sum(line_of_code_decl_list) / len(line_of_code_decl_list)})
        # 2.3.6
        class_metrics.update({'AvgLineCodeExe': sum(line_of_code_exe_list) / len(line_of_code_exe_list)})

        # 2.3.7
        class_metrics.update({'MaxLineCode': max(line_of_code_list)})
        # 2.3.8
        class_metrics.update({'MaxLineCodeDecl': max(line_of_code_decl_list)})
        # 2.3.9
        class_metrics.update({'MaxLineCodeExe': max(line_of_code_exe_list)})

        # 2.3.10
        class_metrics.update({'MinLineCode': min(line_of_code_list)})
        # 2.3.11
        class_metrics.update({'MinLineCodeDecl': min(line_of_code_decl_list)})
        # 2.3.12
        class_metrics.update({'MinLineCodeExe': min(line_of_code_exe_list)})

        # 2.3.13
        class_metrics.update({'SDLineCode': np.std(line_of_code_list)})
        # 2.3.14
        class_metrics.update({'SDLineCodeDecl': np.std(line_of_code_decl_list)})
        # 2.3.15
        class_metrics.update({'SDLineCodeExe': np.std(line_of_code_exe_list)})

        class_metrics.update({'LogLineCode': math.log10(sum(line_of_code_list)+1)})
        class_metrics.update({'LogLineCodeDecl': math.log10(sum(line_of_code_decl_list)+1)})
        class_metrics.update({'LogLineCodeExe': math.log10(sum(line_of_code_exe_list)+1)})

        # CSLOC_NAMM
        # 2.3.16
        class_metrics.update({'CountLineCodeNAMM': sum(line_of_code_namm_list)})
        # 2.3.17
        class_metrics.update({'CountLineCodeDeclNAMM': sum(line_of_code_decl_namm_list)})

        # print('!@#', sum(line_of_code_decl_namm_list))
        # quit()

        # 2.3.18
        class_metrics.update({'CountLineCodeExeNAMM': sum(line_of_code_exe_namm_list)})

        # 2.3.19
        class_metrics.update({'AvgLineCodeNAMM': sum(line_of_code_namm_list) / len(line_of_code_namm_list)})
        # 2.3.20
        class_metrics.update(
            {'AvgLineCodeDeclNAMM': sum(line_of_code_decl_namm_list) / len(line_of_code_decl_namm_list)})
        # 2.3.21
        class_metrics.update({'AvgLineCodeExeNAMM': sum(line_of_code_exe_namm_list) / len(line_of_code_exe_namm_list)})

        # 2.3.22
        class_metrics.update({'MaxLineCodeNAMM': max(line_of_code_namm_list)})
        # 2.3.23
        class_metrics.update({'MaxLineCodeDeclNAMM': max(line_of_code_decl_namm_list)})
        # 2.3.24
        class_metrics.update({'MaxLineCodeExeNAMM': max(line_of_code_exe_namm_list)})

        # 2.3.25
        class_metrics.update({'MinLineCodeNAMM': min(line_of_code_namm_list)})
        # 2.3.26
        class_metrics.update({'MinLineCodeDeclNAMM': min(line_of_code_decl_namm_list)})
        # 2.3.27
        class_metrics.update({'MinLineCodeExeNAMM': min(line_of_code_exe_namm_list)})

        # 2.3.28
        class_metrics.update({'SDLineCodeNAMM': np.std(line_of_code_namm_list)})
        # 2.3.29
        class_metrics.update({'SDLineCodeDeclNAMM': np.std(line_of_code_decl_namm_list)})
        # print('!@#', np.std(line_of_code_decl_namm_list))
        # quit()
        # 2.3.30
        class_metrics.update({'SDLineCodeExeNAMM': np.std(line_of_code_exe_namm_list)})

        # ----------------------------------------------------------------
        # 2.4 CSNOST (3-->30)
        # To be completed in future work
        number_of_stmt_list = list()
        number_of_stmt_namm_list = list()

        number_of_stmt_decl_list = list()
        number_of_stmt_decl_namm_list = list()

        number_of_stmt_exe_list = list()
        number_of_stmt_exe_namm_list = list()

        for method in method_list:
            number_of_stmt_list.append(method.metric(['CountStmt'])['CountStmt'])
            number_of_stmt_decl_list.append(method.metric(['CountStmtDecl'])['CountStmtDecl'])
            number_of_stmt_exe_list.append(method.metric(['CountStmtExe'])['CountStmtExe'])
            if not j_code_odor_metric.is_accesor_or_mutator(input_method=method):
                number_of_stmt_namm_list.append(method.metric(['CountStmt'])['CountStmt'])
                number_of_stmt_decl_namm_list.append(method.metric(['CountStmtDecl'])['CountStmtDecl'])
                number_of_stmt_exe_namm_list.append(method.metric(['CountStmtExe'])['CountStmtExe'])

        cls.remove_none_from_lists([number_of_stmt_list, number_of_stmt_namm_list,
                                    number_of_stmt_decl_list, number_of_stmt_decl_namm_list,
                                    number_of_stmt_exe_list, number_of_stmt_exe_namm_list])

        # CSNOST_All
        # 2.4.4
        class_metrics.update({'AvgStmt': sum(number_of_stmt_list) / len(number_of_stmt_list)})
        # 2.4.5
        class_metrics.update({'AvgStmtDecl': sum(number_of_stmt_decl_list) / len(number_of_stmt_decl_list)})
        # 2.4.6
        class_metrics.update({'AvgStmtExe': sum(number_of_stmt_exe_list) / len(number_of_stmt_exe_list)})

        # 2.4.7
        class_metrics.update({'MaxStmt': max(number_of_stmt_list)})
        # 2.4.8
        class_metrics.update({'MaxStmtDecl': max(number_of_stmt_decl_list)})
        # 2.4.9
        class_metrics.update({'MaxStmtExe': max(number_of_stmt_exe_list)})

        # 2.4.10
        class_metrics.update({'MinStmt': min(number_of_stmt_list)})
        # 2.4.11
        class_metrics.update({'MinStmtDecl': min(number_of_stmt_decl_list)})
        # 2.4.12
        class_metrics.update({'MinStmtExe': min(number_of_stmt_exe_list)})

        # 2.4.13
        class_metrics.update({'SDStmt': np.std(number_of_stmt_list)})
        # 2.4.14
        class_metrics.update({'SDStmtDecl': np.std(number_of_stmt_decl_list)})
        # 2.4.15
        class_metrics.update({'SDStmtExe': np.std(number_of_stmt_exe_list)})

        class_metrics.update({'LogStmt': math.log10(sum(number_of_stmt_list)+1)})
        class_metrics.update({'LogStmtDecl': math.log10(sum(number_of_stmt_decl_list)+1)})
        class_metrics.update({'LogStmtExe': math.log10(sum(number_of_stmt_exe_list)+1)})

        # CSNOST_NAMM
        # 2.4.16
        class_metrics.update({'CountStmtNAMM': sum(number_of_stmt_namm_list)})
        # 2.4.17
        class_metrics.update({'CountStmtDeclNAMM': sum(number_of_stmt_decl_namm_list)})
        # 2.4.18
        class_metrics.update({'CountStmtExeNAMM': sum(number_of_stmt_exe_namm_list)})

        # 2.4.19
        class_metrics.update({'AvgStmtNAMM': sum(number_of_stmt_namm_list) / len(number_of_stmt_namm_list)})
        # 2.4.20
        class_metrics.update(
            {'AvgStmtDeclNAMM': sum(number_of_stmt_decl_namm_list) / len(number_of_stmt_decl_namm_list)})
        # 2.4.21
        class_metrics.update({'AvgStmtExeNAMM': sum(number_of_stmt_exe_namm_list) / len(number_of_stmt_exe_namm_list)})

        # 2.4.22
        class_metrics.update({'MaxStmtNAMM': max(number_of_stmt_namm_list)})
        # 2.4.23
        class_metrics.update({'MaxStmtDeclNAMM': max(number_of_stmt_decl_namm_list)})
        # 2.4.24
        class_metrics.update({'MaxStmtExeNAMM': max(number_of_stmt_exe_namm_list)})

        # 2.4.25
        class_metrics.update({'MinStmtNAMM': min(number_of_stmt_namm_list)})
        # 2.4.26
        class_metrics.update({'MinStmtDeclNAMM': min(number_of_stmt_decl_namm_list)})
        # 2.4.27
        class_metrics.update({'MinStmtExeNAMM': min(number_of_stmt_exe_namm_list)})

        # 2.4.28
        class_metrics.update({'SDStmtNAMM': np.std(number_of_stmt_namm_list)})
        # 2.4.29
        class_metrics.update({'SDStmtDeclNAMM': np.std(number_of_stmt_decl_namm_list)})
        # 2.4.30
        class_metrics.update({'SDStmtExeNAMM': np.std(number_of_stmt_exe_namm_list)})

        # Class number of not accessor or mutator methods
        # Class max_nesting (4)
        CSNOMNAMM = 0
        max_nesting_list = list()
        for method in method_list:
            max_nesting_list.append(method.metric(['MaxNesting'])['MaxNesting'])
            if not j_code_odor_metric.is_accesor_or_mutator(input_method=method):
                CSNOMNAMM += 1

        cls.remove_none_from_lists([max_nesting_list])

        class_metrics.update({'CSNOMNAMM': CSNOMNAMM})

        class_metrics.update({'MinNesting': min(max_nesting_list)})
        class_metrics.update({'AvgNesting': sum(max_nesting_list) / len(max_nesting_list)})
        class_metrics.update({'SDNesting': np.std(max_nesting_list)})

        # Custom (JCodeOdor) coupling metrics
        class_metrics.update({'RFC': j_code_odor_metric.rfc(input_class=entity)})
        class_metrics.update({'FANIN': j_code_odor_metric.FANIN(db=db, class_entity=entity)})
        class_metrics.update({'FANOUT': j_code_odor_metric.FANOUT(db=db, class_entity=entity)})
        class_metrics.update({'ATFD': UnderstandUtility.ATFD(db=db, class_entity=entity)})  ### not implement

        class_metrics.update({'CFNAMM': j_code_odor_metric.CFNAMM_Class(class_name=entity)})
        class_metrics.update({'DAC': UnderstandUtility.get_data_abstraction_coupling(db=db, class_entity=entity)})
        class_metrics.update({'NumberOfMethodCalls': UnderstandUtility.number_of_method_call(class_entity=entity)})

        # Visibility metrics
        # Understand built-in metrics plus one custom metric.
        class_metrics.update({'CSNOAMM': j_code_odor_metric.NOMAMM(class_entity=entity)})

        # Inheritance metrics
        class_metrics.update({'NIM': j_code_odor_metric.NIM(class_name=entity)})
        class_metrics.update({'NMO': j_code_odor_metric.nmo(input_class=entity)})

        class_metrics.update({'NOII': UnderstandUtility.NOII(db=db)})  # Not implemented

        # ---------------------------------------
        # New added metric (version 0.3.0, dataset 0.5.0)
        class_count_path_list = list()
        class_count_path_log_list = list()
        class_knots_list = list()
        for method in method_list:
            class_count_path_list.append(method.metric(['CountPath'])['CountPath'])
            class_count_path_log_list.append(method.metric(['CountPathLog'])['CountPathLog'])
            class_knots_list.append(method.metric(['Knots'])['Knots'])

        cls.remove_none_from_lists([class_count_path_list, class_count_path_log_list, class_knots_list])

        class_metrics.update({'SumCountPath': sum(class_count_path_list)})
        class_metrics.update({'MinCountPath': min(class_count_path_list)})
        class_metrics.update({'MaxCountPath': max(class_count_path_list)})
        class_metrics.update({'AvgCountPath': sum(class_count_path_list)/len(class_count_path_list)})
        class_metrics.update({'SDCountPath': np.std(class_count_path_list)})

        class_metrics.update({'SumCountPathLog': sum(class_count_path_log_list)})
        class_metrics.update({'MinCountPathLog': min(class_count_path_log_list)})
        class_metrics.update({'MaxCountPathLog': max(class_count_path_log_list)})
        class_metrics.update({'AvgCountPathLog': sum(class_count_path_log_list) / len(class_count_path_log_list)})
        class_metrics.update({'SDCountPathLog': np.std(class_count_path_log_list)})

        class_metrics.update({'SumKnots': sum(class_knots_list)})
        class_metrics.update({'MinKnots': min(class_knots_list)})
        class_metrics.update({'MaxKnots': max(class_knots_list)})
        class_metrics.update({'AvgKnots': sum(class_knots_list) / len(class_knots_list)})
        class_metrics.update({'SDKnots': np.std(class_knots_list)})

        constructor = UnderstandUtility.get_constructor_of_class_java(db=db, class_name=entity.longname(),class_entity=entity)
        class_metrics.update({'NumberOfClassConstructors': len(constructor)})

        class_metrics.update({'NumberOfDepends': len(entity.depends())})
        class_metrics.update({'NumberOfDependsBy': len(entity.dependsby())})

        class_metrics.update({'NumberOfClassInItsFile': len(UnderstandUtility.get_number_of_class_in_file_java(db=db, class_entity=entity))})

        return class_metrics


    def compute_java_class_metrics_lexicon(cls, db=None, entity=None):
        """

        :param db:
        :param entity:
        :return:
        """
        class_lexicon_metrics_dict = dict()

        # for ib in entity.ib():
        #     print('entity ib', ib)

        # Compute lexicons
        tokens_list = list()
        identifiers_list = list()
        keywords_list = list()
        operators_list = list()

        return_and_print_count = 0
        return_and_print_kw_list = ['return', 'print', 'printf', 'println', 'write', 'writeln']

        condition_count = 0
        condition_kw_list = ['if', 'for', 'while', 'switch', '?', 'assert', ]

        uncondition_count = 0
        uncondition_kw_list = ['break', 'continue',]

        exception_count = 0
        exception_kw_list = ['try', 'catch', 'throw', 'throws', 'finally', ]

        new_count = 0
        new_count_kw_list = ['new']

        super_count = 0
        super_count_kw_list = ['super']

        dots_count = 0

        try:
            #print(entity)
            #print('ec', entity.parent().id())
            source_file_entity = db.ent_from_id(entity.id())

            #print('file', type(source_file_entity), source_file_entity.longname())
            #print(entity)

            for lexeme in entity.lexer(show_inactive=False):
                print(lexeme.text(), ': ', lexeme.token())
                tokens_list.append(lexeme.text())
                if lexeme.token() == 'Identifier':
                    identifiers_list.append(lexeme.text())
                if lexeme.token() == 'Keyword':
                    keywords_list.append(lexeme.text())
                if lexeme.token() == 'Operator':
                    operators_list.append(lexeme.text())
                if lexeme.text() in return_and_print_kw_list:
                    return_and_print_count += 1
                if lexeme.text() in condition_kw_list:
                    condition_count += 1
                if lexeme.text() in uncondition_kw_list:
                    uncondition_count += 1
                if lexeme.text() in exception_kw_list:
                    exception_count += 1
                if lexeme.text() in new_count_kw_list:
                    new_count += 1
                if lexeme.text() in super_count_kw_list:
                    super_count += 1
                if lexeme.text() == '.':
                    dots_count += 1
        except:
            raise RuntimeError('Error in computing class lexical metrics for class "{0}"'.format(entity.longname()))

        number_of_assignments = operators_list.count('=')
        number_of_operators_without_assignments = len(operators_list) - number_of_assignments
        number_of_unique_operators = len(set(list(filter('='.__ne__, operators_list))))

        class_lexicon_metrics_dict.update({'NumberOfTokens': len(tokens_list)})
        class_lexicon_metrics_dict.update({'NumberOfUniqueTokens': len(set(tokens_list))})

        class_lexicon_metrics_dict.update({'NumberOfIdentifies': len(identifiers_list)})
        class_lexicon_metrics_dict.update({'NumberOfUniqueIdentifiers': len(set(identifiers_list))})

        class_lexicon_metrics_dict.update({'NumberOfKeywords': len(keywords_list)})
        class_lexicon_metrics_dict.update({'NumberOfUniqueKeywords': len(set(keywords_list))})

        class_lexicon_metrics_dict.update(
            {'NumberOfOperatorsWithoutAssignments': number_of_operators_without_assignments})
        class_lexicon_metrics_dict.update({'NumberOfAssignments': number_of_assignments})
        class_lexicon_metrics_dict.update({'NumberOfUniqueOperators': number_of_unique_operators})

        class_lexicon_metrics_dict.update({'NumberOfDots': dots_count})
        class_lexicon_metrics_dict.update({'NumberOfSemicolons': entity.metric(['CountSemicolon'])['CountSemicolon']})

        class_lexicon_metrics_dict.update({'NumberOfReturnAndPrintStatements': return_and_print_count})
        class_lexicon_metrics_dict.update({'NumberOfConditionalJumpStatements': condition_count})
        class_lexicon_metrics_dict.update({'NumberOfUnConditionalJumpStatements': uncondition_count})
        class_lexicon_metrics_dict.update({'NumberOfExceptionStatements': exception_count})
        class_lexicon_metrics_dict.update({'NumberOfNewStatements': new_count})
        class_lexicon_metrics_dict.update({'NumberOfSuperStatements': super_count})

        # print('Class lexicon metrics:', class_lexicon_metrics_dict)
        return class_lexicon_metrics_dict


    def compute_java_package_metrics(cls, db=None, class_name: str = None):

        # Find package: strategy 2: Dominated strategy
        class_name_list = class_name.split('.')[:-1]
        package_name = '.'.join(class_name_list)
        #print('package_name string', package_name)
        #print(class_name)
        #print(db)
        #print(db.lookup(class_name + '$', 'Package'))
        #print(db.lookup(class_name + '$', 'class').library())
        #print(db.lookup(class_name ))
        package_list = list()
        count=1

        while(len(package_list)==0):
            print('compute_java_package_metrics' + class_name)
            print(class_name_list)
            class_name_list = class_name.split('.')[:-count]
            #print(class_name_list)
            package_name = '.'.join(class_name_list)
            #print(package_name)
            package_list = db.lookup(package_name + '$', 'Package')
            count=count+1
        #print('$$$$$$$$$$$$')
        #print(package_list)
        if package_list is None:
            return None
        if len(package_list) == 0:  # if len != 1 return None!
            return None
        package = package_list[0]
        # print('kind:', package.kind())
        #print('Computing package metrics for class: "{0}" in package: "{1}"'.format(class_name, package.longname()))

        # Print info
        #print('package metrics')
        #print('package metrics11')
        package_metrics = package.metric(package.metrics())
        #print('package metrics222')
        #print('number of metrics:', len(metrics), metrics)
        # for i, metric in enumerate(metrics.keys()):
        #     print(i + 1, ': ', metric, metrics[metric])

        # print('class metrics')
        # metrics2 = entity.metric(entity.metrics())
        # print('number of metrics:', len(metrics), metrics2)
        # for i, metric2 in enumerate(metrics.keys()):
        #     print(i + 1, ': ', metric2, metrics[metric2])

        #
        # print(package.refs('Definein'))
        # for defin in package.refs('Definein'):
        #     print('kind', defin.ent().kind())
        # print(defin, '-->', defin.ent().ents('Java Define', 'Class'))
        # metrics = entity.metric(defin.ent().metrics())
        # print('number of metrics in file:', len(metrics), metrics)
        # for i, metric in enumerate(metrics.keys()):
        #     print(i + 1, ': ', metric, metrics[metric])

        classes_and_interfaces_list = UnderstandUtility.get_package_clasess_java(package_entity=package)
        # print(classes_and_interfaces_list)
        # quit()

        # 2. Custom package metrics
        # 2.1. PKLOC (15)
        pk_loc_list = list()
        pk_loc_decl_list = list()
        pk_loc_exe_list = list()
        for type_entity in classes_and_interfaces_list:
            pk_loc_list.append(type_entity.metric(['CountLineCode'])['CountLineCode'])
            pk_loc_decl_list.append(type_entity.metric(['CountLineCodeDecl'])['CountLineCodeDecl'])
            pk_loc_exe_list.append(type_entity.metric(['CountLineCodeExe'])['CountLineCodeExe'])
        #print('*****')
        cls.remove_none_from_lists([pk_loc_list,  pk_loc_decl_list, pk_loc_exe_list])


        try:
            package_metrics.update({'AvgLineCodeDecl': sum(pk_loc_decl_list) / len(pk_loc_decl_list)})
            package_metrics.update({'AvgLineCodeExe': sum(pk_loc_exe_list) / len(pk_loc_exe_list)})

            package_metrics.update({'MaxLineCode': max(pk_loc_list)})
            package_metrics.update({'MaxLineCodeDecl': max(pk_loc_decl_list)})
            package_metrics.update({'MaxLineCodeExe': max(pk_loc_exe_list)})

            package_metrics.update({'MinLineCode': min(pk_loc_list)})
            package_metrics.update({'MinLineCodeDecl': min(pk_loc_decl_list)})
            package_metrics.update({'MinLineCodeExe': min(pk_loc_exe_list)})

            package_metrics.update({'SDLineCode': np.std(pk_loc_list)})
            package_metrics.update({'SDLineCodeDecl': np.std(pk_loc_decl_list)})
            package_metrics.update({'SDLineCodeExe': np.std(pk_loc_exe_list)})
        except:
            raise TypeError('Error happen when compute packege metric for class "{0}" and list "{1}"'.format(class_name, pk_loc_decl_list))

        # 2.2 PKNOS (15)
        pk_stmt_list = list()
        pk_stmt_decl_list = list()
        pk_stmt_exe_list = list()
        for type_entity in classes_and_interfaces_list:
            pk_stmt_list.append(type_entity.metric(['CountStmt'])['CountStmt'])
            pk_stmt_decl_list.append(type_entity.metric(['CountStmtDecl'])['CountStmtDecl'])
            pk_stmt_exe_list.append(type_entity.metric(['CountStmtExe'])['CountStmtExe'])
        #print('*****')
        cls.remove_none_from_lists([pk_stmt_list, pk_stmt_decl_list, pk_stmt_exe_list])
        #print('*****')
        package_metrics.update({'AvgStmt': sum(pk_stmt_decl_list) / len(pk_stmt_decl_list)})
        package_metrics.update({'AvgStmtDecl': sum(pk_stmt_decl_list) / len(pk_stmt_decl_list)})
        package_metrics.update({'AvgStmtExe': sum(pk_stmt_exe_list) / len(pk_stmt_exe_list)})

        package_metrics.update({'MaxStmt': max(pk_stmt_list)})
        package_metrics.update({'MaxStmtDecl': max(pk_stmt_decl_list)})
        package_metrics.update({'MaxStmtExe': max(pk_stmt_exe_list)})

        package_metrics.update({'MinStmt': min(pk_stmt_list)})
        package_metrics.update({'MinStmtDecl': min(pk_stmt_decl_list)})
        package_metrics.update({'MinStmtExe': min(pk_stmt_exe_list)})

        package_metrics.update({'SDStmt': np.std(pk_stmt_list)})
        package_metrics.update({'SDStmtDecl': np.std(pk_stmt_decl_list)})
        package_metrics.update({'SDStmtExe': np.std(pk_stmt_exe_list)})

        # 2.3 PKCC (20)
        pk_cyclomatic_list = list()
        pk_cyclomatic_namm_list = list()

        pk_cyclomatic_strict_list = list()
        pk_cyclomatic_strict_namm_list = list()

        pk_cyclomatic_modified_list = list()
        pk_cyclomatic_modified_namm_list = list()

        pk_essential_list = list()
        pk_essential_namm_list = list()

        for type_entity in classes_and_interfaces_list:
            pk_cyclomatic_list.append(type_entity.metric(['SumCyclomatic'])['SumCyclomatic'])
            pk_cyclomatic_modified_list.append(type_entity.metric(['SumCyclomaticModified'])['SumCyclomaticModified'])
            pk_cyclomatic_strict_list.append(type_entity.metric(['SumCyclomaticStrict'])['SumCyclomaticStrict'])
            pk_essential_list.append(type_entity.metric(['SumEssential'])['SumEssential'])

        cls.remove_none_from_lists([pk_cyclomatic_list, pk_cyclomatic_strict_list, pk_cyclomatic_modified_list, pk_essential_list])

        package_metrics.update({'MinCyclomatic': min(pk_cyclomatic_list)})
        package_metrics.update({'MinCyclomaticModified': min(pk_cyclomatic_modified_list)})
        package_metrics.update({'MinCyclomaticStrict': min(pk_cyclomatic_strict_list)})
        package_metrics.update({'MinEssential': min(pk_essential_list)})

        package_metrics.update({'SDCyclomatic': np.std(pk_cyclomatic_list)})
        package_metrics.update({'SDCyclomaticModified': np.std(pk_cyclomatic_modified_list)})
        package_metrics.update({'SDCyclomaticStrict': np.std(pk_cyclomatic_strict_list)})
        package_metrics.update({'SDEssential': np.std(pk_essential_list)})

        # 2.4 PKNESTING (4)
        pk_nesting_list = list()
        for type_entity in classes_and_interfaces_list:
            pk_nesting_list.append(type_entity.metric(['MaxNesting'])['MaxNesting'])

        cls.remove_none_from_lists([pk_nesting_list])

        package_metrics.update({'MinNesting': min(pk_nesting_list)})
        package_metrics.update({'AvgNesting': sum(pk_nesting_list) / len(pk_nesting_list)})
        package_metrics.update({'SDNesting': np.std(pk_nesting_list)})

        # 2.5
        # Other Size/Count metrics (understand built-in metrics)

        # PKNOMNAMM: Package number of not accessor or mutator methods
        j_code_odor = JCodeOdorMetric()
        pk_not_accessor_and_mutator_methods_list = list()
        pk_accessor_and_mutator_methods_list = list()
        for type_entity in classes_and_interfaces_list:
            pk_not_accessor_and_mutator_methods_list.append(j_code_odor.NOMNAMM(type_entity))
            pk_accessor_and_mutator_methods_list.append(j_code_odor.NOMAMM(type_entity))

        cls.remove_none_from_lists([pk_not_accessor_and_mutator_methods_list, pk_accessor_and_mutator_methods_list])
        #print('&&&&&&&&&&&&&&&&&&&&')
        #print(classes_and_interfaces_list)
        package_metrics.update({'PKNOMNAMM': sum(pk_not_accessor_and_mutator_methods_list)})

        # 2.6 Visibility metrics
        # Other Visibility metrics metrics (understand built-in metrics)
        package_metrics.update({'PKNOAMM': sum(pk_accessor_and_mutator_methods_list)})
        # To add other visibility metrics

        # 2.7 Inheritance metrics
        package_metrics.update({'PKNOI': len(UnderstandUtility.get_package_interfaces_java(package_entity=package))})
        package_metrics.update(
            {'PKNOAC': len(UnderstandUtility.get_package_abstract_class_java(package_entity=package))})

        # print(len(package_metrics))
        # print(package_metrics)
        return package_metrics


    def compute_java_project_metrics(cls, db):
        print('start compute_java_project_metrics')
        project_metrics = db.metric(db.metrics())
        print(project_metrics)
        # print('number of metrics:', len(project_metrics),  project_metrics)
        # for i, metric in enumerate( project_metrics.keys()):
        #     print(i + 1, ': ',  metric,  project_metrics[metric])

        # print(project_metrics)  # Print Understand built-in metrics

        # 2 Custom project metrics
        print('*******')
        files = UnderstandUtility.get_project_files_java(db=db)
        print('*******')
        # 2.1 PJLOC (30)
        pj_loc_list = list()
        pj_loc_decl_list = list()
        pj_loc_exe_list = list()

        pj_stmt_list = list()
        pj_stmt_decl_list = list()
        pj_stmt_exe_list = list()

        for file_entity in files:
            pj_loc_list.append(file_entity.metric(['CountLineCode'])['CountLineCode'])
            pj_loc_decl_list.append(file_entity.metric(['CountLineCodeDecl'])['CountLineCodeDecl'])
            pj_loc_exe_list.append(file_entity.metric(['CountLineCodeExe'])['CountLineCodeExe'])

            pj_stmt_list.append(file_entity.metric(['CountStmt'])['CountStmt'])
            pj_stmt_decl_list.append(file_entity.metric(['CountStmtDecl'])['CountStmtDecl'])
            pj_stmt_exe_list.append(file_entity.metric(['CountStmtExe'])['CountStmtExe'])

        cls.remove_none_from_lists([pj_loc_list, pj_loc_decl_list, pj_loc_exe_list,
                                    pj_stmt_list, pj_stmt_decl_list,  pj_stmt_exe_list])

        project_metrics.update({'AvgLineCodeDecl': sum(pj_loc_decl_list) / len(pj_loc_decl_list)})
        project_metrics.update({'AvgLineCodeExe': sum(pj_loc_exe_list) / len(pj_loc_exe_list)})

        project_metrics.update({'MaxLineCode': max(pj_loc_list)})
        project_metrics.update({'MaxLineCodeDecl': max(pj_loc_decl_list)})
        project_metrics.update({'MaxLineCodeExe': max(pj_loc_exe_list)})

        project_metrics.update({'MinLineCode': min(pj_loc_list)})
        project_metrics.update({'MinLineCodeDecl': min(pj_loc_decl_list)})
        project_metrics.update({'MinLineCodeExe': min(pj_loc_exe_list)})

        project_metrics.update({'SDLineCode': np.std(pj_loc_list)})
        project_metrics.update({'SDLineCodeDecl': np.std(pj_loc_decl_list)})
        project_metrics.update({'SDLineCodeExe': np.std(pj_loc_exe_list)})

        # 2.2. PJNOST (15)
        project_metrics.update({'AvgStmt': sum(pj_stmt_list) / len(pj_stmt_list)})
        project_metrics.update({'AvgStmtDecl': sum(pj_stmt_decl_list) / len(pj_stmt_decl_list)})
        project_metrics.update({'AvgStmtExe': sum(pj_stmt_exe_list) / len(pj_stmt_exe_list)})

        project_metrics.update({'MaxStmt': max(pj_stmt_list)})
        project_metrics.update({'MaxStmtDecl': max(pj_stmt_decl_list)})
        project_metrics.update({'MaxStmtExe': max(pj_stmt_exe_list)})

        project_metrics.update({'MinStmt': min(pj_stmt_list)})
        project_metrics.update({'MinStmtDecl': min(pj_stmt_decl_list)})
        project_metrics.update({'MinStmtExe': min(pj_stmt_exe_list)})

        project_metrics.update({'SDStmt': np.std(pj_stmt_list)})
        project_metrics.update({'SDStmtDecl': np.std(pj_stmt_decl_list)})
        project_metrics.update({'SDStmtExe': np.std(pj_stmt_exe_list)})
        #print('sss')
        # 2.3 Other Count/Size metrics
        packages = db.ents('Java Package')
        #print('yy')
        print('number of packages', len(packages))
        project_metrics.update({'NumberOfPackages': len(packages)})

        j_code_odor = JCodeOdorMetric()
        pj_number_of_method_namm = 0
        #print('*******')
        for class_ in UnderstandUtility.get_project_classes_java(db=db):
            pj_number_of_method_namm += j_code_odor.NOMNAMM(class_)
        project_metrics.update({'PJNOMNAMM': pj_number_of_method_namm})

        # 2.4 PJCC (20): Project cyclomatic complexity
        pj_cyclomatic_list = list()
        pj_cyclomatic_namm_list = list()

        pj_cyclomatic_strict_list = list()
        pj_cyclomatic_strict_namm_list = list()

        pj_cyclomatic_modified_list = list()
        pj_cyclomatic_modified_namm_list = list()

        pj_essential_list = list()
        pj_essential_namm_list = list()

        for type_entity in files:
            pj_cyclomatic_list.append(type_entity.metric(['SumCyclomatic'])['SumCyclomatic'])
            pj_cyclomatic_modified_list.append(type_entity.metric(['SumCyclomaticModified'])['SumCyclomaticModified'])
            pj_cyclomatic_strict_list.append(type_entity.metric(['SumCyclomaticStrict'])['SumCyclomaticStrict'])
            pj_essential_list.append(type_entity.metric(['SumEssential'])['SumEssential'])

        cls.remove_none_from_lists([pj_cyclomatic_list, pj_cyclomatic_strict_list,
                                    pj_cyclomatic_modified_list, pj_essential_list ])

        project_metrics.update({'SumCyclomatic': sum(pj_cyclomatic_list)})
        project_metrics.update({'SumCyclomaticModified': sum(pj_cyclomatic_modified_list)})
        project_metrics.update({'SumCyclomaticStrict': sum(pj_cyclomatic_strict_list)})
        project_metrics.update({'SumEssential': sum(pj_essential_list)})

        project_metrics.update({'MaxCyclomatic': max(pj_cyclomatic_list)})
        project_metrics.update({'MaxCyclomaticModified': max(pj_cyclomatic_modified_list)})
        project_metrics.update({'MaxCyclomaticStrict': max(pj_cyclomatic_strict_list)})
        project_metrics.update({'MaxEssential': max(pj_essential_list)})

        project_metrics.update({'AvgCyclomatic': sum(pj_cyclomatic_list) / len(pj_cyclomatic_list)})
        project_metrics.update(
            {'AvgCyclomaticModified': sum(pj_cyclomatic_modified_list) / len(pj_cyclomatic_modified_list)})
        project_metrics.update({'AvgCyclomaticStrict': sum(pj_cyclomatic_strict_list) / len(pj_cyclomatic_strict_list)})
        project_metrics.update({'AvgEssential': sum(pj_essential_list) / len(pj_essential_list)})

        project_metrics.update({'MinCyclomatic': min(pj_cyclomatic_list)})
        project_metrics.update({'MinCyclomaticModified': min(pj_cyclomatic_modified_list)})
        project_metrics.update({'MinCyclomaticStrict': min(pj_cyclomatic_strict_list)})
        project_metrics.update({'MinEssential': min(pj_essential_list)})

        project_metrics.update({'SDCyclomatic': np.std(pj_cyclomatic_list)})
        project_metrics.update({'SDCyclomaticModified': np.std(pj_cyclomatic_modified_list)})
        project_metrics.update({'SDCyclomaticStrict': np.std(pj_cyclomatic_strict_list)})
        project_metrics.update({'SDEssential': np.std(pj_essential_list)})

        # 2.4 PKNESTING (4)
        pj_nesting_list = list()
        for type_entity in files:
            pj_nesting_list.append(type_entity.metric(['MaxNesting'])['MaxNesting'])

        cls.remove_none_from_lists([pj_nesting_list])

        project_metrics.update({'MinNesting': min(pj_nesting_list)})
        project_metrics.update({'AvgNesting': sum(pj_nesting_list) / len(pj_nesting_list)})
        project_metrics.update({'SDNesting': np.std(pj_nesting_list)})

        # 3 Inheritance metrics
        project_metrics.update({'PJNOI': len(UnderstandUtility.get_project_interfaces_java(db=db))})
        project_metrics.update({'PJNAC': len(UnderstandUtility.get_project_abstract_classes_java(db=db))})

        return project_metrics


    def remove_none_from_lists(cls, lists: list = None):
        for i, list_ in enumerate(lists):
            if len(list_) == 0:
                list_.append(0)
                warnings.warn('Empty list passed!')
                #raise ValueError('Empty list passed!')
            # else:
            #     list_ = [i for i in list_ if i is not None]
            #     if len(list_) == 0:
            #         list_.append(0)
            #         raise ValueError('Required data for systematic metric computation is not enough!')


#db_path = r'C:/Users/moham/Desktop/New folder/'
#os.chdir(db_path)
#db = understand.open('2dxgujun__AndroidTagGroup.udb')
#classes_list = db.ents('Java Class ~Interface ~Enum ~Unknown ~Unresolved ~Jar ~Library')

#Calculate_Metrics().extract_metrics_and_coverage_all(r'C:/Users/moham/Desktop/New folder/',classes_list,r'C:/Users/moham/Desktop/New folder/')


obj = Calculate_Metrics()
obj.main()

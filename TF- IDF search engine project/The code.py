#Import the necessary libraries   
import pandas as pd
import numpy as np
import os
import math
from numpy import dot
from numpy.linalg import norm
import scipy.stats as ss
import fnmatch

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

#This function to compute the value  of term frequency  
def computeTF(x, y):
    tf = {}
    Counter = len(y)
    for word, count in x.items():
        tf[word] = count/float(Counter)
    return tf
#This function to compute the value of inverse document frequency
  
def computeIDF(documents , unique_words):
   idf={}
   N=len(documents)
   for i in unique_words:
     count=0
     for sen in documents:
       if i in sen.split():
         count=count+1
       idf[i]=(math.log10((1+N)/(count+1)))
   return idf


#This function to compute both TF and IDF
def computeTFIDF(tf, idf):
    tfidf = {}
    for word, val in tf.items():
        tfidf[word] = val*idf[word]
    return tfidf

# Allow the user to choose  one out of three options 
n = int((input('''Please enter  1,2 or 3 
            \n1 : Find the similarity between your query and the documents in the default database "20_newsgroup" 
            \n2: Compare the similarity between the documents in your database
            \n3: Compere the similarity between your query and your database 
            \nenter your selection : ''')))
if n == 1:
    #take the uesr query
    the_query = input("Enter the query you want to search for  : ")
    directory = r"C:\20_newsgroups"
    list_of_filename=['the query']
    df=[the_query]
    file_location =['the query']         
    result=set()
    q=set(the_query.split(" "))
    result.update(q)
    #loop over the directory and read the files in it 
    for path,dirs,files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename,"*"):
                with open(os.path.join(path, filename),'r')as f:
                    p = os.path.join(path,filename)
                    list_of_filename.append(filename)
                    file_location .append(path)
                    docments = f.read()
                    z=set(docments.split(" "))
                    df.append(docments)
                    result.update(z)
                    continue
            else:
                    continue
                 
    df1 =pd.DataFrame()
    idf=[]
    tff=[]
    final=[]
    df_tfidf=pd.DataFrame()
    tf_df=pd.DataFrame()
    IDF_values=computeIDF(df,result)
    for i in df :
        x=i.split(" ")
        wordDict = dict.fromkeys(result, 0)
        for word in x:
            wordDict[word]+=1
        idf.append(wordDict) 
        tf=computeTF(wordDict, x)
        tff.append(tf)
        df1=df1.append(wordDict,ignore_index=True)
        tf_df=tf_df.append(tf,ignore_index=True)
        tfidf=computeTFIDF(tf,IDF_values)
        tfidf1=list(tfidf.values())
        final.append(tfidf1)
        df_tfidf=df_tfidf.append(tfidf,ignore_index=True)
    cos_rank=[]
    cos=[]         
    for i in final:
        for j in final:
            cos_sim = dot(i, j)/(norm(i)*norm(j))
            cos.append(cos_sim)
    r=final[0]    
    for x in final:
        cos_sim = dot(x,r)/(norm(x)*norm(r))
        cos_rank.append(cos_sim) 
    rank=(len(cos_rank)+1) - ss.rankdata(cos_rank).astype(int)                          
    the_rank  = pd.DataFrame({
                    'file name': list_of_filename,
                    'similarity with the query': cos_rank,
                    'rank': rank,
                    'file location ':file_location 
                    })
                
    cosine_similarity_matrix= np.array(cos).reshape(len(cos_rank),len(cos_rank) ) 
    print(' The cosine_similarity_matrix :\n')
    print(cosine_similarity_matrix)
    df_tfidfal_df= the_rank.sort_values(by=['rank'])
    print(df_tfidfal_df)      
                
                      
                    
elif n==2:
    directory = input("Enter the  dictatory of  the database you want compare the similarity between its document   : ")
    list_of_filename=[]
    df=[]
    file_location =[]         
    result=set()
    for path,dirs,files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename,"*"):
                with open(os.path.join(path, filename),'r')as f:
                    p = os.path.join(path,filename)
                    list_of_filename.append(filename)
                    file_location .append(path)
                    docments = f.read()
                    z=set(docments.split(" "))
                    df.append(docments)
                    result.update(z)
                    continue
            else:
                    continue
                 
    df1 =pd.DataFrame()
    idf=[]
    tff=[]
    final=[]
    df_tfidf=pd.DataFrame()
    tf_df=pd.DataFrame()
    IDF_values=computeIDF(df,result)
    for i in df :
        x=i.split(" ")
        wordDict = dict.fromkeys(result, 0)
        for word in x:
            wordDict[word]+=1
        idf.append(wordDict) 
        tf=computeTF(wordDict, x)
        tff.append(tf)
        df1=df1.append(wordDict,ignore_index=True)
        tf_df=tf_df.append(tf,ignore_index=True)
        tfidf=computeTFIDF(tf,IDF_values)
        tfidf1=list(tfidf.values())
        final.append(tfidf1)
        df_tfidf=df_tfidf.append(tfidf,ignore_index=True)
    cos=[]
             
    for i in final:
        for j in final:
            cos_sim = dot(i, j)/(norm(i)*norm(j))
            cos.append(cos_sim)
    cos = np.array(cos).reshape(len(list_of_filename),len(list_of_filename))
    print(' The cosine_similarity_matrix :\n')       
    print(cos)
    the_fiels  = pd.DataFrame({
                    'file name': list_of_filename,
                    'file location':file_location 
                    })
    print(the_fiels)

elif n==3:
    the_query = input("Enter your query : ")
    directory = input("Enter the dictatory of the database you want to search in  : ")
    list_of_filename=['the query']
    df=[the_query]
    file_location =['the query']         
    result=set()
    q=set(the_query.split(" "))
    result.update(q)
    for path,dirs,files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename,"*"):
                with open(os.path.join(path, filename),'r')as f:
                    p = os.path.join(path,filename)
                    list_of_filename.append(filename)
                    file_location .append(path)
                    docments = f.read()
                    z=set(docments.split(" "))
                    df.append(docments)
                    result.update(z)
                    continue
            else:
                    continue
                 
    df1 =pd.DataFrame()
    idf=[]
    tff=[]
    final=[]
    df_tfidf=pd.DataFrame()
    tf_df=pd.DataFrame()
    IDF_values=computeIDF(df,result)
    for i in df :
        x=i.split(" ")
        wordDict = dict.fromkeys(result, 0)
        for word in x:
            wordDict[word]+=1
        idf.append(wordDict) 
        tf=computeTF(wordDict, x)
        tff.append(tf)
        df1=df1.append(wordDict,ignore_index=True)
        tf_df=tf_df.append(tf,ignore_index=True)
        tfidf=computeTFIDF(tf,IDF_values)
        tfidf1=list(tfidf.values())
        final.append(tfidf1)
        df_tfidf=df_tfidf.append(tfidf,ignore_index=True)
    cos_rank=[]
    cos=[]
                
    for i in final:
        for j in final:
            cos_sim = dot(i, j)/(norm(i)*norm(j))
            cos.append(cos_sim)
    r=final[0]    
    for x in final:
        cos_sim = dot(x,r)/(norm(x)*norm(r))
        cos_rank.append(cos_sim) 
    rank=(len(cos_rank)+1) - ss.rankdata(cos_rank).astype(int)                          
    the_rank  = pd.DataFrame({
                    'file name': list_of_filename,
                    'similarity with the query': cos_rank,
                    'rank': rank,
                    'file location':file_location 
                    })
                
    cosine_similarity_matrix= np.array(cos).reshape(len(cos_rank),len(cos_rank))
    print(' The cosine_similarity_matrix :\n')       
    print(cosine_similarity_matrix)
    df_tfidfal_df= the_rank.sort_values(by=['rank'])
    print(df_tfidfal_df)

else:
    print("Invalid input, pleas try again ")

    

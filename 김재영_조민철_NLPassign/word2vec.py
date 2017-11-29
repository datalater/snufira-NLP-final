import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

import time
import gensim
from konlpy.tag import Mecab
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


text = []
mecab = Mecab()


# Data Cleansing

for news_name in ['chosun_full', 'donga_full', 'hani_full', 'joongang_full', 'kh_full'] :
    with open('./Data/news/'+ news_name + '.txt', 'r') as f:
        lines = f.read()
        sentences = lines.split('.')
        for i in range(len(sentences)):
            dic = mecab.morphs(sentences[i])
            dicSize = len(dic)
                        
            for idx in range(dicSize-1):
                if(idx==dicSize):
                        break
                
                else:
                    if(dic[idx]=='새' and dic[(idx+1)]=='정치') :
                        dic.remove('새')
                        dic.remove('정치')
                        dic.insert(idx,"새정치")
                        dicSize = dicSize-1
            
            text.append(dic)

            
            
# Make Model - cbow & skipgram

start_time = time.time() 

word2vecCbow= Word2Vec(text, min_count=15, size = 300, sg=0, iter= 10) #sg=0 : cbow
word2vecSkip = Word2Vec(text, min_count=15, size = 300, sg=1, iter= 10) #sg=1 : skip gram
print("word2vec success")

vectorsCbow = word2vecCbow.wv
vectorsSkip = word2vecSkip.wv
print("wv making success")


vectorsCbow.save('word2vecCbow_300_10.bin')
vectorsSkip.save('word2vecSkip_300_10.bin')
print("model save success")


print("Elapsed time: %s seconds" %(time.time() - start_time))


# Model load
word2vecCbow = KeyedVectors.load('word2vecCbow_300_10.bin')
word2vecSkip = KeyedVectors.load('word2vecSkip_300_10.bin')



name_list = ["박근혜","오바마","김정은","아베","청와대","백악관","새누리당","최순실"]
words_list = [["서울","일본","한국"],["서울","미국","한국"],["서울","북한","한국"],
                ["박근혜","일본","한국"],["박근혜","미국","한국"],["박근혜","북한","한국"],
                ["새누리당","미국","한국"],["새정치","미국","한국"],["청와대","미국","한국"],["민주주의","북한","한국"],
                ["김무성","새정치","새누리당"],["보수","새정치","새누리당"]]


# word2vec CBOW result check

print("\nword2vec_CBOW - Word Similarity\n")

for name in name_list:
    name = name.encode("utf-8")
    w = word2vecCbow.most_similar(unicode(name,"utf-8"), topn=1)
    w1 = w[0][0]
    w2 = w[0][1]
    print "[%s]'s most similar: [%s]" %(name, w1),
    print(w2)

print("\nword2vec_CBOW - Word Analogy\n")
for words in words_list:
    a= word2vecCbow.most_similar(positive=[unicode(words[0],"utf-8"),unicode(words[1],"utf-8")], negative=[unicode(words[2],"utf-8")], topn=1)
    print ("[%s+%s-%s] corresponds to [%s]" % (words[0], words[1], words[2], a[0][0].encode("utf-8")))


    
# word2vec SKIPGRAM result check
    
print("\nword2vec_SKIPGRAM - Word Similarity\n")
for name in name_list:
    name = name.encode("utf-8")
    w = word2vecSkip.most_similar(unicode(name,"utf-8"), topn=1)
    w1 = w[0][0]
    w2 = w[0][1]
    print "[%s]'s most similar: [%s]" %(name, w1),
    print(w2)
    
    
print("\nword2vec_SKIPGRAM - Word Analogy\n")
for words in words_list:
    a= word2vecSkip.most_similar(positive=[unicode(words[0],"utf-8"),unicode(words[1],"utf-8")], negative=[unicode(words[2],"utf-8")], topn=1)
    print ("[%s+%s-%s] corresponds to [%s]" % (words[0], words[1], words[2], a[0][0].encode("utf-8")))

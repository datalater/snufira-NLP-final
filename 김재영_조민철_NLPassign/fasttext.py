import fasttext
import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

from konlpy.tag import Mecab
from gensim.models import KeyedVectors

# Data Cleansing
mecab = Mecab()
articlesMorphs = open("articlesMorphs.txt","w") 

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
            
            for word in dic:
                articlesMorphs.write("%s " % word)
                
                
                
articlesMorphs.close()


# fasttext model making
cbow = fasttext.cbow("articlesMorphs.txt","fastCbow_300_10",dim=300,epoch=10)
print("fastCbow model saved!!!")
skipgram = fasttext.skipgram("articlesMorphs.txt","fastSkip_300_10",dim=300,epoch=10)
print("fastSkip model saved!!!")


# load model.
fastCbow = KeyedVectors.load_word2vec_format('fastCbow_300_10.vec')
print("fastCbow model loaded")
fastSkip = KeyedVectors.load_word2vec_format('fastSkip_300_10.vec')
print("fastSkip model loaded")


name_list = ["박근혜","오바마","김정은","아베","청와대","백악관","새누리당","최순실"]
words_list = [["서울","일본","한국"],["서울","미국","한국"],["서울","북한","한국"],
                ["박근혜","일본","한국"],["박근혜","미국","한국"],["박근혜","북한","한국"],
                ["새누리당","미국","한국"],["새정치","미국","한국"],["청와대","미국","한국"],["민주주의","북한","한국"],
                ["김무성","새정치","새누리당"],["보수","새정치","새누리당"]]


# fasttext CBOW result check

print("\nfastText_CBOW - Word Similarity\n")

for name in name_list:
    name = name.encode("utf-8")
    w = fastCbow.most_similar(unicode(name,"utf-8"), topn=1)
    w1 = w[0][0]
    w2 = w[0][1]
    print "[%s]'s most similar: [%s]" %(name, w1),
    print(w2)

print("\nfastText_CBOW - Word Analogy\n")
for words in words_list:
    a= fastCbow.most_similar(positive=[unicode(words[0],"utf-8"),unicode(words[1],"utf-8")], negative=[unicode(words[2],"utf-8")], topn=1)
    print ("[%s+%s-%s] corresponds to [%s]" % (words[0], words[1], words[2], a[0][0].encode("utf-8")))


    
# fasttext SKIPGRAM result check
    
print("\nfastText_SKIPGRAM - Word Similarity\n")
for name in name_list:
    name = name.encode("utf-8")
    w = fastSkip.most_similar(unicode(name,"utf-8"), topn=1)
    w1 = w[0][0]
    w2 = w[0][1]
    print "[%s]'s most similar: [%s]" %(name, w1),
    print(w2)
    
    
print("\nfastText_SKIPGRAM - Word Analogy\n")
for words in words_list:
    a= fastSkip.most_similar(positive=[unicode(words[0],"utf-8"),unicode(words[1],"utf-8")], negative=[unicode(words[2],"utf-8")], topn=1)
    print ("[%s+%s-%s] corresponds to [%s]" % (words[0], words[1], words[2], a[0][0].encode("utf-8")))

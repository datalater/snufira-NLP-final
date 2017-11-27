# -*- coding: utf-8 -*- 

'''

author@hpshin
한국어 형태소 분석 모듈 for Word2Vec
konlpy를 사용하고 Twitter/Mecab형태소 분석기 사용
원래 컴파일시  유니코드 아스키 에러가 나기 때문에 
앞의 sys와 reload(sys)로 해결

'''

import codecs

import sys
reload(sys)
sys.setdefaultencoding('utf8')

#from konlpy.tag import Twitter

from konlpy.tag import Mecab

tagger = Mecab()

# 각자의 한글 데이터 파일을 열고 형태소 분석 후 쓸 파일 정함
corpusIn =codecs.open('corpusAllNews.txt', 'rU', encoding='utf-8')
corpusOut = codecs.open('corpusAllNewsNoTagNoStop.txt','w', encoding='utf-8')

#단어/tag형태로 출력하는 함수
def getTag(content):
        return ["{}/{}".format(word, tag) for word, tag in tagger.pos(content)]
   
for line in corpusIn:
	word = line.split()
	
	for num in word:
		corpusOut.write(' '.join(getTag(num))+ ' ')
        corpusOut.write('\n')



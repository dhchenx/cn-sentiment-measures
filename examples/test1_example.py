from cn_sentiment_measures.sentiment_measures import SentimentMeasures

sm = SentimentMeasures()

word_list=["我","今天","很","高兴"]

print("Here are measure values with normal measures!")
print('APD = ',sm.APD(word_list))
print('RPD = ',sm.RPD(word_list))
print('LS = ',sm.LS(word_list))

print()
print("Here are measure values optimized by using degree words!")
print('APD* = ',sm.APD_with_degree(word_list))
print('RPD* = ',sm.RPD_with_degree(word_list))
print('LS* = ',sm.LS_with_degree(word_list))
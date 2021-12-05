# Chinese Sentiment Measures 

A toolkit for estimating Chinese sentiment scores with multiple measures. 

```pip
pip install cn-sentiment-measures
```

## Measures
1. Absolute Proportional Difference - APD. Bounds: [-1, 1]
2. Relative Proportional Difference - RPD. Bounds: [-1, 1]
3. Logit scale - LC.  Bounds: [-infinity, +infinity]

Improved versions of the above measures by integrating degree words are marked with `_with_degree` in the toolkit. 

## Examples
```python

from cn_sentiment_measures.sentiment_measures import SentimentMeasures

sm = SentimentMeasures()

word_list=["我","今天","很","高兴"]

print("Here are measure values with normal measures!")
print('APD = ',sm.APD(word_list))
print('RPD = ',sm.RPD(word_list))
print('LS = ',sm.LS(word_list))

print()
print("Here are measure values optimized by using adverb words!")
print('APD* = ',sm.APD_with_degree(word_list))
print('RPD* = ',sm.RPD_with_degree(word_list))
print('LS* = ',sm.LS_with_degree(word_list))

```

## License
The project is provided by [Donghua Chen](https://github.com/dhchenx). 


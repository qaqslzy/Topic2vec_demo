# Topic2Vec
根据Topic2Vec: Learning Distributed Representations of Topics这篇论文，用tensorflow实现的Topic2vec
依赖于gensim，nltk，tensorflow
## topic2vec
用tensorflow来进行topic2vec的运算
## PreProcess
对数据进行预处理，词型还原停用词和提取词干
## readmodle
用tensorflow运算生成的模型，来画出散点图

## 其他
- 对tensorflow还是不是很熟悉，菜渣写码，难受。
- 我用的数据是一个list，然后list里的每个元素都是dict，dict里面的"abstract"，是运算所要用到的数据，也就是一段文本，想跑的可以改一下**topic2vec.py**和**PreProcess.py**里那部分的相应的代码。

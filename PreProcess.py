from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import string
import json
import os
import nltk


def preprocess(open_porterp_temmer=False):
    with open("tingyongci.txt") as f:
        stoplist = f.readlines()

    for i in range(len(stoplist)):
        stoplist[i] = stoplist[i].strip()

    # 词形还原
    wnl = WordNetLemmatizer()
    # 词干提取
    ps = PorterStemmer()

    with open("results_2865_full.json") as f:
        data = json.load(f)

    for item in data:
        item["abstract"] = item["abstract"].encode("utf-8").decode("utf-8")
        doc = item["abstract"].lower()

        # 去标点
        for c in string.punctuation:
            doc = doc.replace(c, " ")
        # 去数字
        for c in string.digits:
            doc = doc.replace(c, " ")

        doc = nltk.word_tokenize(doc)
        clean_doc = []
        for word in doc:
            if len(word) >= 3 and word not in stoplist and wordnet.synsets(word):
                word = wnl.lemmatize(word)  # 词形还原
                if open_porterp_temmer:
                    word = ps.stem(word)  # 词干提取
                clean_doc.append(word)

        item["abstract"] = ' '.join(clean_doc)

    with open("change_ps_{}.json".format(open_porterp_temmer), "w") as f:
        json.dump(data, f, indent=4)

    return data


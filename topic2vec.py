import tensorflow as tf
import numpy as np
import json
import collections
from gensim import corpora, models
from six.moves import xrange
import random
from PreProcess import preprocess
import os

# settings
num_skips = 8
skip_window = 5
batch_size = 256
embedding_size = 128  # 向量纬度
num_sampled = 64  # nce的负采样

open_ps = False  # 是否提取词干

if not os.path.exists("change_ps_{}.json".format(open_ps)):
    data_all = preprocess(open_ps)
else:
    # 读取数据
    with open("change_ps_{}.json".format(open_ps)) as f:
        data_all = json.load(f)


# 单词
words = []
# 句子
sentcences = []

# TODO 做PreProcess
for item in data_all:
    words.extend(item["abstract"].strip().split())
    sentcences.append(item["abstract"].strip())

# 确定主题的数量
NUM_TOPIC = 15

# 关于验证的一些参数
valid_size = 15  # 验证大小
valid_window = NUM_TOPIC  # 验证总体
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


texts = [[word for word in document.lower().split()] for document in sentcences]
# 词典
gensim_dictionary = corpora.Dictionary(texts)

# 词库，以(词，词频)方式存贮
gensim_corpus = [gensim_dictionary.doc2bow(text) for text in texts]

# lda
ldamodel = models.LdaModel(gensim_corpus, id2word=gensim_dictionary, num_topics=NUM_TOPIC,per_word_topics=True,minimum_probability=0.000001)


reverse_topic_dictionary = dict(zip(gensim_dictionary.values(),gensim_dictionary.keys()))

# 找到每个词对应的topic
word2topic = {}
for item in gensim_dictionary.items():
    t = sorted(ldamodel.get_term_topics(item[1]), key=lambda x: x[0],
               reverse=True)
    try:
        t = t[0][0]
    except:
        word2topic[item[0]] = 0
    else:
        word2topic[item[0]] = t


# 通过单词的id找到对应topic的id
def get_topic_id(word_id):
    """
    :param word_id: 单词对应的id
    :return: 对应的主题id
    """
    word = reverse_dictionary[word_id]
    gensim_word_id = reverse_topic_dictionary[word.lower()]
    return word2topic[gensim_word_id]


vocabulary_size = 40000


# 构建数据集
def bulid_dataset(sentcences):
    """
    :param sentcences: 所有句子
    :return: sent_data 句子， count 单词和词频，
             dictionary单词对应编号，reverse_dictionary 编号对应单词
    """
    count = [["UNK",-1]]
    #  统计词频
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    # 编号
    for word, _ in count:
        dictionary[word] = len(dictionary)
    unk_count = 0
    sent_data = []
    for sentcence in sentcences:
        data = []
        for word in sentcence.split():
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data.append(index)

        sent_data.append(data)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))

    return sent_data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = bulid_dataset(sentcences)


if vocabulary_size > len(reverse_dictionary):
    vocabulary_size = len(reverse_dictionary)

# 更新valid的对应id
for i, v in enumerate(valid_examples):
    valid_examples[i] = v + vocabulary_size


word_data = []
for sent in data:
    word_data += sent


# 删除数据节省内存
del data_all

data_index = 0

# skip-gram model 的 扫描器
def generate_batch(batch_size, num_skips, skip_window):
    """
    :param batch_size: 扫描块的大小
    :param num_skips: 输入数字的重用次数
    :param skip_window: 上下文的大小
    :return: trains and labels
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    newbatch = np.ndarray(shape=(batch_size * 2), dtype=np.int32)

    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    newlabels = np.ndarray(shape=(batch_size * 2, 1), dtype=np.int32)
    # 扫描的窗口 上下文+中间词
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    # 存储词所使用的缓存，大小为窗口那么大
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(word_data):
        data_index = 0
    # 将词加进缓存中
    buffer.extend(word_data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        # 上下文
        context_words = [w for w in range(span) if w != skip_window]
        # 从上下文种挑选num_skips个作为应为的预测值
        words_to_use = random.sample(context_words, num_skips)
        # 将上面选出的中间词和应为的预测值放入batch和labels数组中
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            # topic[i * num_skips + j] = get_topic_id(buffer[skip_window])
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(word_data):
            buffer.extend(word_data[0:span])
            data_index = span
        else:
            buffer.append(word_data[data_index])
            data_index += 1
    # 防止在结尾处漏掉单词
    data_index = (data_index + len(word_data) - span) % len(word_data)

    # 加入topic
    for i, v in enumerate(batch):
        newbatch[i * 2 + 1] = vocabulary_size + get_topic_id(v)
        newbatch[i * 2] = v

    for i, v in enumerate(labels):
        newlabels[i * 2 + 1] = v
        newlabels[i * 2] = v

    return newbatch, newlabels


graph = tf.Graph()

with graph.as_default():
    train_word_data = tf.placeholder(tf.int32,shape=[batch_size * 2])
    train_labels = tf.placeholder(tf.int32,shape=[batch_size * 2, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    words_embeddings = tf.Variable(tf.random_uniform([vocabulary_size + NUM_TOPIC, embedding_size],-1.0,1.0))

    # weight and  biases
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size + NUM_TOPIC, embedding_size],
                                                  stddev=1.0 / np.sqrt(embedding_size)))

    nce_biases = tf.Variable(tf.zeros([vocabulary_size + NUM_TOPIC]))

    # input
    embed_word = tf.nn.embedding_lookup(words_embeddings, train_word_data)

    # loss
    loss = tf.reduce_mean(tf.nn.nce_loss(
        nce_weights, nce_biases, train_labels,
        embed_word, num_sampled, vocabulary_size + NUM_TOPIC
    ))

    # train
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(words_embeddings), 1, keep_dims=True))
    normalized_embeddings = words_embeddings / norm

    # 通过topic和word的余弦来计算相似度
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

cfg = tf.ConfigProto(allow_soft_placement=True)
cfg.gpu_options.allow_growth = True

# 训练步数
num_step = 60000

with tf.Session(config=cfg,graph=graph) as sess:

    step_delta = num_step // 20
    init.run()
    print("start")
    average_loss = 0
    for step in range(num_step):
        batch_word_data, batch_labels = generate_batch(batch_size,num_skips= num_skips,skip_window = skip_window)
        feed_dict = {
            train_word_data:batch_word_data,
            train_labels:batch_labels
        }
        _, step_loss = sess.run([optimizer, loss], feed_dict= feed_dict)
        average_loss += step_loss
        if step % step_delta == 0:
            if step > 0:
                average_loss = average_loss / step_delta
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0

    final_word_embeddings = normalized_embeddings.eval()

    tf.add_to_collection('final_word_embeddings',normalized_embeddings)
    tf.add_to_collection("similarity",similarity)

    # 计算相似度
    topicsword = {}
    sim = similarity.eval()
    for i in xrange(valid_size):
        valid_topic = valid_examples[i] - vocabulary_size
        top_k = 20  # 相似最大的前top_k
        nearest = (-sim[i, :]).argsort()[1+NUM_TOPIC:top_k + 1 + NUM_TOPIC]
        topicsword[valid_topic] = []
        print(valid_topic, ":", end=" ")
        for k in xrange(top_k):
            topicsword[valid_topic].append(nearest[k])
            print(reverse_dictionary.get(nearest[k], "topic_%d" % (nearest[k] - vocabulary_size)), ":",
                  sim[i, nearest[k]], end=" ")
        print()

    # 保存模型
    try:
        import os
        import argparse
        import sys

        current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--log_dir',
            type=str,
            default=os.path.join(current_path, 'log'),
            help='The log directory for TensorBoard summaries.')
        FLAGS, unparsed = parser.parse_known_args()

        saver.save(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))
    except:
        print("save error")

    # 保存字典
    with open("reverse_dictionary.json", "w") as f:
        json.dump(reverse_dictionary, f, indent=4)


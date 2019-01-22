import tensorflow as tf
import os
import sys
import traceback
import matplotlib.pyplot as plt


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
log_path = os.path.join(current_path, 'log')

NUM_TOPIC = 15
meta_path =  log_path + "/model.ckpt.meta"
model_path = log_path + "/model.ckpt"
saver = tf.train.import_meta_graph(meta_path)


# 画图
def plot_by_list(low_dim_embs_list, filename='tsne_all.png'):
    plt.figure(figsize=(18, 18))

    color = ["b", "c", "g", "k",
             "m", "r", "orange", "y"]
    markers = ["o", ",", "^", ".", "v", "<", ">"]

    global color_idx, markers_idx

    for low_dim_embs in low_dim_embs_list:
        for i in low_dim_embs:
            x, y = i[:]
            plt.scatter(x, y, c=color[color_idx], s=60, marker=markers[markers_idx])
        if (color_idx + 1) % 8 == 0:
            color_idx = 0
            markers_idx += 1
        else:
            color_idx += 1

    plt.savefig(filename)


# 读取模型里的数据然后画出图
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    saver.restore(sess, model_path) # 导入变量值
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    normalized_embeddings = graph.get_collection("final_word_embeddings")[0]
    similarity = graph.get_collection("similarity")[0]
    sim = similarity.eval()
    final_word_embeddings = normalized_embeddings.eval()
    topicsword = {}
    for i in range(15):
        valid_topic = i
        top_k = 50
        nearest = (-sim[i, :]).argsort()[1 + NUM_TOPIC:top_k + 1 + NUM_TOPIC]
        topicsword[valid_topic] = []
        for k in range(top_k):
            topicsword[valid_topic].append(nearest[k])

    color_idx = 0
    markers_idx = 0

    try:
        # 降维
        from sklearn.manifold import TSNE, MDS

        tsne = TSNE(perplexity=45, n_components=2, n_iter=8000, learning_rate=10)

        showtopic_word = {}
        low_list = []
        for k, v in topicsword.items():
            low = tsne.fit_transform(final_word_embeddings[v, :])
            low_list.append(low)
            # plot_with_labels(low, "topic_model%d" % k)
            # if k % 8 == 0:
            #     color_idx = 0
            #     markers_idx += 1
            # else:
            #     color_idx += 1
        plot_by_list(low_list[:8])
    except ImportError:
        traceback.print_exc()
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")


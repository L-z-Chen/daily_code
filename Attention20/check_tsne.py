import tqdm
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_stream import *
from backbone import *

sess_config = tf.ConfigProto(allow_soft_placement=True)

sess_config.gpu_options.allow_growth = True
sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=sess_config)

model_fn = resnet_50
num_gpus = 1

# data
DATA_STREAM = Cifar_datastream(sess, 100 * num_gpus, False)
val_img, val_label = DATA_STREAM.get_test_batch()

logits, probs, end_point = model_fn(val_img, is_train=False)

sess.run(tf.global_variables_initializer())
saver_save = tf.train.Saver(tf.global_variables())


MAIN_ITER = 500

# saver_save.restore(sess, "models/cifar_chirality.ckpt")
saver_save.restore(sess, "models/cifar_com.ckpt")


L_out = []
label_out = []

for i in tqdm.trange(MAIN_ITER):
    L, P, tmp_label = sess.run([logits, probs, tf.argmax(val_label, axis=1)])
    L_out.extend(P)
    label_out.extend(tmp_label)


tsne = TSNE(n_components=2, verbose=10, n_iter=3000)
tsne.fit_transform(np.asarray(L_out))
print(tsne.embedding_)



import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.keys())

color = [mcolors.TABLEAU_COLORS[colors[label_out[i]]] for i in range(len(L_out))]

plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=color, alpha=1.0)

plt.show()

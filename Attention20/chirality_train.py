import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import tqdm
import matplotlib.pyplot as plt
import numpy as np
import collections
import six
import itertools
from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config
import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2,4,6"

from Loss_Block import *
from data_stream import *

sess_config = tf.ConfigProto(allow_soft_placement=True)

sess_config.gpu_options.allow_growth = True
sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=sess_config)

import nets.resnet as resnet


def resnet50(image, is_train=True, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)
    image = tf.reshape(preprocessed, [-1, 224, 224, 3])
    arg_scope = resnet.resnet_utils.resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_point = resnet.resnet_v2_50(image, 1001, is_training=is_train, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


# config
model_fn = resnet50
num_gpus = 4
MAIN_ITER = 30000
# TEST_Iter_NUM = 391
lr_rate = 5e-7
global_s = tf.Variable(0, trainable=False, name='gs')
# TEST_Iter_NUM = 100
SAVE_PATH = "/data/zhaoxian/Chirality/models/resnet_v2_50_2020_10_17/resnet_v2_50.ckpt"
# RESTORE_PATH = ""
RESTORE_PATH = "/data/zhaoxian/Chirality/models/resnet_v2_50_2020_10_17/resnet_v2_50.ckpt"

# data


DATA_STREAM = ImageNet_datastream(sess, 32 * num_gpus)
train_img, train_label = DATA_STREAM.get_train_batch()

tower_losses = []
tower_gradvars = []


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']
        # ps_ops = ['Variable', 'VariableV2', 'VarHandleOp', 'Const', 'Fill', 'Assign', 'Identity', 'ApplyAdam']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string('/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


with tf.device("/cpu:0"):
    split_train_img = tf.split(train_img, num_gpus)
    split_train_label = tf.split(train_label, num_gpus)
    update_ops = 0
    for i in range(num_gpus):
        worker_device = '/{}:{}'.format('gpu', i)
        device_setter = local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
        with tf.device(device_setter):
            Loss_DIC = Get_ALL_Loss(model_fn, split_train_img[i], split_train_label[i])

            Loss = Loss_DIC['closs']
            Loss += 10*Loss_DIC['chirality_loss']
            Loss += Loss_DIC['l2_loss']

            model_params = tf.trainable_variables()
            grads = tf.gradients(Loss, model_params)
            tower_gradvars.append(zip(grads, model_params))
            tower_losses.append(Loss)



    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
        for var, grads in six.iteritems(all_grads):
            # Average gradients on the same device as the variables
            # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
            gradvars.append((avg_grad, var))

# Device that runs the ops to apply global gradient updates.
consolidation_device = '/cpu:0'
with tf.device(consolidation_device):
    optimizer = slim.train.AdamOptimizer(learning_rate=lr_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(gradvars, global_step=global_s)
    Loss = tf.reduce_mean(tower_losses)

# eval
val_img, val_label = DATA_STREAM.get_test_batch()

with tf.device("/cpu:0"):
    tower_acc = []
    tower_val_acc = []

    val_split_img = tf.split(val_img, num_gpus)
    val_split_label = tf.split(val_label, num_gpus)
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpus):
            worker_device = '/{}:{}'.format('gpu', i)
            device_setter = local_device_setter(
                ps_device_type='gpu',
                worker_device=worker_device,
                ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                    num_gpus, tf.contrib.training.byte_size_load_fn))

            with tf.device(device_setter):
                # eval
                logits, probs, end_point = model_fn(val_split_img[i], is_train=False)
                correct_p = tf.equal(tf.argmax(logits, 1), (tf.argmax(val_split_label[i], 1)))
                tmp_accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))
                tower_acc.append(tmp_accuracy)
    val_clean_accuracy = tf.reduce_mean(tower_acc)

# merged = tf.summary.merge_all()
merged = tf.summary.merge([
    tf.summary.scalar('accuracy', Loss_DIC['clean_acc']),
    tf.summary.scalar('loss', Loss_DIC['closs']),
    # tf.summary.scalar('ch', Loss_DIC['chirality_loss']),
    tf.summary.scalar('l2', Loss_DIC['l2_loss']),
    # tf.summary.scalar('l2', Loss_DIC['l2_loss']),
    tf.summary.scalar('val_acc', val_clean_accuracy)
])


if MAIN_ITER != 0:
    SUMMARY_DIR = './log'
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

sess.run(tf.global_variables_initializer())

saver_save_global = tf.train.Saver(tf.global_variables())

if RESTORE_PATH != "":
    if RESTORE_PATH == "/data/zhaoxian/Chirality/models/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt":
        var_load = tf.contrib.framework.get_variables_to_restore(exclude=['beta1_power', 'beta2_power', '.*/Adam', 'gs'])
        saver_save = tf.train.Saver(var_load)
        saver_save.restore(sess, RESTORE_PATH)
    else:
        var_load = tf.contrib.framework.get_variables_to_restore(
            exclude=['beta1_power', 'beta2_power','.*/Adam','gs'])
        saver_save = tf.train.Saver(var_load)
        saver_save.restore(sess, RESTORE_PATH)

sess.graph.finalize()

train_bar = tqdm.trange(MAIN_ITER)
for i in train_bar:
    A, _, tmp_cl,  tmo_l2, gs = sess.run(
        [Loss_DIC['clean_acc'], train_op, Loss_DIC['closs'],  Loss_DIC['l2_loss'], global_s])
    print(gs)
    if i % 10 == 0:
        summary = sess.run(merged)
        summary_writer.add_summary(summary, gs)
    train_bar.set_description(str([A, tmp_cl, tmo_l2]))

    if i % 10000 == 0 and SAVE_PATH != "":
        saver_save_global.save(sess, SAVE_PATH)

saver_save_global.save(sess, SAVE_PATH)

# #
# Final_acc = 0
# pbar = tqdm.trange(TEST_Iter_NUM)
# for i in pbar:
#     tmp_acc = sess.run(val_clean_accuracy)
#     Final_acc += tmp_acc
#     pbar.set_description("clean_Final_acc:{:.2f}".format(Final_acc / (i + 1) * 100))
#
# print("clean_Final_acc:{:.2f}".format(Final_acc / TEST_Iter_NUM * 100))

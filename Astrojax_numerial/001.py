import tensorflow as tf
import numpy as np
import numba as nb
from numba import jit
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1' # 设置GPU设备

# 获取当前工作路径
a = os.getcwd()
print(a)

# x = tf.Variable(initial_value=[1.,-1.])
# with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
#     y = x**2
# y_grad = tape.gradient(y, x)        # 计算y关于x的导数
# print(y, y_grad)


# variables
m1 = 0.025
m2 = m1
# h1
# h2
R0 = 0.5
# R1
# R2
# w
# T
a = 0.2
b = 0.4
g = 9.8
eta = 0.001
# Compute

# Constant:
m1 = tf.constant(m1)
m2 = tf.constant(m2)
g = tf.constant(g)
eta = tf.constant(eta)
# add the inatial known variables：
a = tf.constant(a)
b = tf.constant(b)
R0 = tf.constant(R0)
# Variables:
h1 = tf.Variable(initial_value=0.)
h2 = tf.Variable(initial_value=0.2)
R1 = tf.Variable(initial_value=0.1)
R2 = tf.Variable(initial_value=0.4)
w = tf.Variable(initial_value=0.4)
T = tf.Variable(initial_value=0.)
variables = [h1, h2, R1, R2, w, T]


# function:
# @nb.jit(nopython=True,parallel=True)
def Astraojax(m1,m2,h1,h2,R0, R1,R2,w,a,b,T,g,eta):
    

    y1 = (h1*T)/tf.sqrt(abs(h1)**2+abs(tf.sqrt(1-a**2)*R1)**2+abs(R0-a*R1)**2)-(m1+m2)*g

    y2 = (h2*T)/tf.sqrt(abs(h2)**2+abs(a*R1-b*R2)**2+abs(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)**2)-m2*g

    y3 = T*(((1-a**2)*R1**2)/(tf.sqrt(abs(a*R1)**2+abs(tf.sqrt(1-a**2)*R1)**2)*tf.sqrt(abs(h1)**2+abs(tf.sqrt(1-a**2)*R1)**2+abs(R0-a*R1)**2))-(a*R1*(R0-a*R1))/(tf.sqrt(abs(a*R1)**2+abs(tf.sqrt(1-a**2)*R1)**2)*tf.sqrt(abs(h1)**2+abs(tf.sqrt(1-a**2)*R1)**2+abs(R0-a*R1)**2)))-T*(-((a*R1*(a*R1-b*R2))/(tf.sqrt(abs(a*R1)**2+abs(tf.sqrt(1-a**2)*R1)**2)*tf.sqrt(abs(h2)**2+abs(a*R1-b*R2)**2+abs(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)**2)))+(tf.sqrt(1-a**2)*R1*(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2))/(tf.sqrt(abs(a*R1)**2+abs(tf.sqrt(1-a**2)*R1)**2)*tf.sqrt(abs(h2)**2+abs(a*R1-b*R2)**2+abs(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)**2)))-(w**2)*m1*R1

    y4 = T*((a*tf.sqrt(1-a**2)*R1**2*w)/(tf.sqrt(abs(h1)**2+abs(tf.sqrt(1-a**2)*R1)**2+abs(R0-a*R1)**2)*tf.sqrt(abs(a*R1*w)**2+abs(tf.sqrt(1-a**2)*R1*w)**2))+(tf.sqrt(1-a**2)*R1*(R0-a*R1)*w)/(tf.sqrt(abs(h1)**2+abs(tf.sqrt(1-a**2)*R1)**2+abs(R0-a*R1)**2)*tf.sqrt(abs(a*R1*w)**2+abs(tf.sqrt(1-a**2)*R1*w)**2)))-T*((tf.sqrt(1-a**2)*R1*(a*R1-b*R2)*w)/(tf.sqrt(abs(h2)**2+abs(a*R1-b*R2)**2+abs(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)**2)*tf.sqrt(abs(a*R1*w)**2+abs(tf.sqrt(1-a**2)*R1*w)**2))+(a*R1*(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)*w)/(tf.sqrt(abs(h2)**2+abs(a*R1-b*R2)**2+abs(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)**2)*tf.sqrt(abs(a*R1*w)**2+abs(tf.sqrt(1-a**2)*R1*w)**2)))-eta*w*R1

    y5 = T*(-((tf.sqrt(1-b**2)*R2*(a*R1-b*R2)*w)/(tf.sqrt(abs(h2)**2+abs(a*R1-b*R2)**2+abs(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)**2)*tf.sqrt(abs(b*R2*w)**2+abs(tf.sqrt(1-b**2)*R2*w)**2)))-(b*R2*(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)*w)/(tf.sqrt(abs(h2)**2+abs(a*R1-b*R2)**2+abs(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)**2)*tf.sqrt(abs(b*R2*w)**2+abs(tf.sqrt(1-b**2)*R2*w)**2)))-eta*w*R2

    y6 = T*(-((b*R2*(a*R1-b*R2))/(tf.sqrt(abs(b*R2)**2+abs(tf.sqrt(1-b**2)*R2)**2)*tf.sqrt(abs(h2)**2+abs(a*R1-b*R2)**2+abs(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)**2)))+(tf.sqrt(1-b**2)*R2*(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2))/(tf.sqrt(abs(b*R2)**2+abs(tf.sqrt(1-b**2)*R2)**2)*tf.sqrt(abs(h2)**2+abs(a*R1-b*R2)**2+abs(-tf.sqrt(1-a**2)*R1+tf.sqrt(1-b**2)*R2)**2)))-m2*R2*w**2
    
    loss = y1**2 + y2**2+ y3**2+ y4**2+ y5**2+ y6**2

    # print(loss)
    
    return loss
# tensorboard:
summary_writer = tf.summary.create_file_writer('./play_test/tensorboard')
log_dir = './play_test/tensorboard'
tf.summary.trace_on(graph=True, profiler=True)  # 开启Trace，可以记录图结构和profile信息
# 进行训练
with summary_writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)    # 保存Trace信息到文件

num_epoch = 10000
learning_rate = 7e-2
optimizer = tf.keras.optimizers.SGD(learning_rate)


for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        # y_pred = a * X + b
        # loss = tf.reduce_sum(tf.square(y_pred - y))
        loss = Astraojax(m1,m2,h1,h2,R0, R1,R2,w,a,b,T,g,eta)
        # print('loss',loss)
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # Record data
    with summary_writer.as_default():                               # 希望使用的记录器
        tf.summary.scalar("loss", loss, step=e)
        # tf.summary.scalar("a", a, step=e)  # 还可以添加其他自定义的变量
    # print('grad',grads)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    # print(tf.print(loss))

tf.print(variables)
print('Final loss = ')
tf.print(loss)
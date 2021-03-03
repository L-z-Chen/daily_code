# import warnings
# warnings.simplefilter('ignore')

import os


from matplotlib import pylab as P

from chirality_train_dir.data_stream import *

import tensorflow as tf
# 1. Install tf_slim using: pip install git+https://github.com/adrianc-a/tf-slim.git@remove_contrib
# 2. Replace imports of slim with import tf_slim as slim
#    in the models/research/slim folder - in inception_v3.py and inception_utils.py.
# import tensorflow.contrib.slim.nets as nets
# from tensorflow.contrib.slim.nets import inception_v3

# From our repository.
import saliency
# import chirality_train_dir.nets.resnet as resnet
import sys
sys.path.append('../')


from chirality_train_dir.data_stream import *
from chirality_train_dir.backbone import *
import argparse
#

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
tf.logging.set_verbosity(tf.logging.ERROR)

slim = tf.contrib.slim
num_gpus=1



print('*************************************************')

# def getargs():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument("is_change_iter", default=1,help="echo the string you use here")
#     # parser.add_argument("iter", default=0,help="echo the string you use here")
#
#     parser.add_argument("no", default=0 ,help="echo the string you use here")
#
#     args = parser.parse_args()
#     return args
#


def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    im = ((im + 1) * 127.5).astype(np.uint8)
    P.imshow(im)
    P.title(title)


def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')

    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)


def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)


def ShowDivergingImage(grad, title='', percentile=99, ax=None):
    if ax is None:
        fig, ax = P.subplots()
    else:
        fig = ax.figure

    P.axis('off')
    divider = saliency.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    P.title(title)



def vanilla_gradient_smoothgrad(graph, sess, y, images):
    print('Vanilla_adn_Smoothgrad')
    gradient_saliency = saliency.GradientSaliency(graph, sess, y, images)

    # Compute the vanilla mask and the smoothed mask.
    vanilla_mask_3d = gradient_saliency.GetMask(im, feed_dict={neuron_selector: prediction_class})
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, feed_dict={neuron_selector: prediction_class})

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
    image_grad.append(vanilla_mask_grayscale)
    image_grad.append(smoothgrad_mask_grayscale)

# Guided Backprop & SmoothGrad
def Guided_Backprop_SmoothGrad(graph, sess, y, images):
    print('Guided_Backprop_SmoothGrad')
    guided_backprop = saliency.GuidedBackprop(graph, sess, y, images)

    # Compute the vanilla mask and the smoothed mask.
    vanilla_guided_backprop_mask_3d = guided_backprop.GetMask(
        im, feed_dict={neuron_selector: prediction_class})
    smoothgrad_guided_backprop_mask_3d = guided_backprop.GetSmoothedMask(
        im, feed_dict={neuron_selector: prediction_class})

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_guided_backprop_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_guided_backprop_mask_3d)
    image_grad.append(vanilla_mask_grayscale)
    image_grad.append(smoothgrad_mask_grayscale)

# Integrated Gradients & SmoothGrad

def Integrated_Gradients_and_SmoothGrad(graph, sess, y, images):
    print('Integrated_Gradients_and_SmoothGrad')
    integrated_gradients = saliency.IntegratedGradients(graph, sess, y, images)

    # Baseline is a black image.
    baseline = np.zeros(im.shape)
    baseline.fill(-1)

    # Compute the vanilla mask and the smoothed mask.
    vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
        im, feed_dict={neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    # Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
    smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
        im, feed_dict={neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)
    image_grad.append(vanilla_mask_grayscale)
    image_grad.append(smoothgrad_mask_grayscale)

# XRAI Full and Fast

def XRAI_Full(graph, sess, y, images):
    print('XRAI_Full')
    xrai_object = saliency.XRAI(graph, sess, y, images)

    # Compute XRAI attributions with default parameters
    xrai_attributions = xrai_object.GetMask(im, feed_dict={neuron_selector: prediction_class})
    image_grad.append(xrai_attributions)
    # Set up matplot lib figures.
    # Show XRAI heatmap attributions

    # Show most salient 30% of the image
    mask = xrai_attributions > np.percentile(xrai_attributions, 70)
    im_mask = np.array(im)
    im_mask[~mask] = 0
    # ShowImage(im_mask, title='Top 30%', ax=P.subplot(ROWS, COLS, 3))
    image_grad.append((im_mask+1)/2.0)

def XRAI_Fast(graph, sess, y, images):
    print('XRAI_Fast')
    xrai_object = saliency.XRAI(graph, sess, y, images)
    xrai_params = saliency.XRAIParameters()
    xrai_params.algorithm = 'fast'

    # Compute XRAI attributions with fast algorithm
    xrai_attributions_fast = xrai_object.GetMask(im, feed_dict={neuron_selector: prediction_class},
                                                 extra_parameters=xrai_params)

    # Set up matplot lib figures.
    # ROWS = 1
    # COLS = 3
    # UPSCALE_FACTOR = 20
    # P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
    image_grad.append(xrai_attributions_fast)
    # Show original image
    # ShowImage(im, title='Original Image', ax=P.subplot(ROWS, COLS, 1))

    # Show XRAI heatmap attributions
    # ShowHeatMap(xrai_attributions_fast, title='XRAI Heatmap', ax=P.subplot(ROWS, COLS, 2))

    # Show most salient 30% of the image
    mask = xrai_attributions_fast > np.percentile(xrai_attributions_fast, 70)
    im_mask = np.array(im)
    im_mask[~mask] = 0
    # ShowImage(im_mask, 'Top 30%', ax=P.subplot(ROWS, COLS, 3))
    image_grad.append((im_mask+1)/2.0)

# Blur IG

def Blur_IG(graph, sess, y, images):
    integrated_gradients = saliency.IntegratedGradients(graph, sess, y, images)
    blur_ig = saliency.BlurIG(graph, sess, y, images)

    # Baseline is a black image for vanilla integrated gradients.
    baseline = np.zeros(im.shape)
    baseline.fill(-1)

    # Compute the vanilla mask and the Blur IG mask.
    vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
        im, feed_dict={neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
    blur_ig_mask_3d = blur_ig.GetMask(
        im, feed_dict={neuron_selector: prediction_class})

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(blur_ig_mask_3d)
    image_grad.append(vanilla_mask_grayscale)
    image_grad.append(blur_ig_mask_grayscale)
    # Set up matplot lib figures.
    # ROWS = 1
    # COLS = 2
    # UPSCALE_FACTOR = 10
    # P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
    #
    # # Render the saliency masks.
    # ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 1))
    # ShowGrayscaleImage(blur_ig_mask_grayscale, title='Blur Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))

def Blur_Smootth_Grad(graph, sess, y, images):

    blur_ig = saliency.BlurIG(graph, sess, y, images)

    # Compute the Blur IG mask and Smoothgrad+BlurIG mask.
    blur_ig_mask_3d = blur_ig.GetMask(im, feed_dict={neuron_selector: prediction_class})
    # Smoothed mask for BlurIG will take a while since we are doing nsamples * nsamples computations.
    smooth_blur_ig_mask_3d = blur_ig.GetSmoothedMask(im, feed_dict={neuron_selector: prediction_class})

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(blur_ig_mask_3d)
    smooth_blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(smooth_blur_ig_mask_3d)
    image_grad.append(blur_ig_mask_grayscale)
    image_grad.append(smooth_blur_ig_mask_grayscale)
    # Set up matplot lib figures.
    # ROWS = 1
    # COLS = 2
    # UPSCALE_FACTOR = 10
    # P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
    #
    # # Render the saliency masks.
    # ShowGrayscaleImage(blur_ig_mask_grayscale, title='Blur Integrated Gradients', ax=P.subplot(ROWS, COLS, 1))
    # ShowGrayscaleImage(smooth_blur_ig_mask_grayscale, title='Smoothgrad Blur IG', ax=P.subplot(ROWS, COLS, 2))

def merge_image(graph, sess, y, images):
    # vanilla_gradient_smoothgrad(graph, sess, y, images)
    # Guided_Backprop_SmoothGrad(graph, sess, y, images)
    Integrated_Gradients_and_SmoothGrad(graph, sess, y, images)
    # XRAI_Full(graph, sess, y, images)
    # XRAI_Fast(graph, sess, y, images)
    # Blur_IG(graph, sess, y, images)
    # Blur_Smootth_Grad(graph, sess, y, images)

def showimg(image,j):
    fig, AX = plt.subplots(4, 3, figsize=(105, 28))

    for i in range(3):
        AX[0][i].imshow(image[i])
        AX[1][i].imshow(image[i+3])
        AX[2][i].imshow(image[i + 6])
        AX[3][i].imshow(image[i + 9])
    # plt.show()
    plt.savefig('./photos/test_imagenet_{}.png'.format(j))

sess1 = tf.InteractiveSession()
DATA_STREAM = ImageNet_datastream(sess1, 200 * num_gpus,imgsize=224)
train_img, train_label = DATA_STREAM.get_test_batch()
model_fn = resnet2_net
sess = tf.Session()

neuron_selector = tf.placeholder(tf.int32)
images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

logits, probs, end_point = model_fn(images, is_train=False)
# arg_scope = resnet.resnet_utils.resnet_arg_scope()
# with slim.arg_scope(arg_scope):
#     logits, end_point = resnet.resnet_v2_18(images, 100, is_training=True, reuse=tf.AUTO_REUSE)
#     print('model done')

y = logits[0][neuron_selector]

prediction = tf.argmax(logits, 1)

saver = tf.train.Saver()

iter=10000//200


if __name__ == '__main__':
    # args=getargs()

    img=sess1.run(train_img)

    num=np.random.randint(0,200)
    # num=int(args.no)
    # print(num)
    im = 2*(img[num]/255-0.5)
    # # ShowImage(im)
    print('image done')
    # image=[]
    image_grad = []

    for s in ["/data/zhaoxian/Chirality/models/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt",
              "/data/zhaoxian/Chirality/models/resnet_v2_50_2020_10_17/resnet_v2_50.ckpt",
              "/data/zhaoxian/Chirality/models/imagenet64_base_2018_06_26.ckpt/imagenet64_base_2018_06_26.ckpt",
              "/data/zhaoxian/Chirality/models/imagenet64_alp025_2018_06_26.ckpt/imagenet64_alp025_2018_06_26.ckpt"
              ]:
        image_grad.append((im+1)/2.0)
            # Restore the checkpoint
        ckpt_file = s

        saver.restore(sess, ckpt_file)
        print('load model done')
        # Construct the scalar neuron tensor.
        # Construct tensor for predictions.

        prediction_class = sess.run(prediction, feed_dict = {images: [im]})[0]
        print("Prediction class: " + str(prediction_class))
        merge_image(sess.graph, sess, y, images)

    showimg(image_grad,num)
# vanilla_gradient_smoothgrad()
# Guided_Backprop_SmoothGrad()
# Integrated_Gradients_and_SmoothGrad()
# XRAI_Full()
# XRAI_Fast()
# Blur_IG()
# Blur_Smootth_Grad()

# Vanilla_adn_Smoothgrad()
# showimg()
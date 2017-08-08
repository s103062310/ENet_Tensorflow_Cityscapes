import tensorflow as tf
import numpy as np
import sys

def preprocess(image, annotation=None, height=360, width=480):
    '''
    Performs preprocessing for one set of image and annotation for feeding into network.
    NO scaling of any sort will be done as per original paper.

    INPUTS:
    - image (Tensor): the image input 3D Tensor of shape [height, width, 3]
    - annotation (Tensor): the annotation input 3D Tensor of shape [height, width, 1]
    - height (int): the output height to reshape the image and annotation into
    - width (int): the output width to reshape the image and annotation into

    OUTPUTS:
    - preprocessed_image(Tensor): the reshaped image tensor
    - preprocessed_annotation(Tensor): the reshaped annotation tensor
    '''

    #Convert the image and annotation dtypes to tf.float32 if needed
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.cast(image, tf.float32)

    image.set_shape(shape=(height, width, 3))
    image = tf.image.resize_images(image, [height, width])
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    image.set_shape(shape=(height, width, 3))

    if not annotation == None:
        annotation.set_shape(shape=(height, width, 1))
        annotation = tf.image.resize_images(annotation, [height, width])
        annotation = tf.image.resize_image_with_crop_or_pad(annotation, height, width)
        annotation.set_shape(shape=(height, width, 1))

        return image, annotation

    return image

def postprocess(prediction, height=360, width=480, dataset="CamVid"):
    '''
    Performs postprocessing for a batch of prediction results.
    From class ID to RGB.
    
    INPUTS:
    - prediction (ndarray): the prediction result of one image with shape [height, width]
    
    OUTPUTS:
    - segmentation (ndarray): RGB of the predition result with shape [height, width, 3]    
    '''
    
    #Define map of class to color
    if dataset=="CamVid":
        class_num = 12
        color_map = [
            #   R    G    B         ID   class
              [128, 128, 128],    #  0   sky
              [128,   0,   0],    #  1   building
              [192, 192, 128],    #  2   pole
              [128,  64, 128],    #  3   road
              [ 60,  40, 222],    #  4   pavement
              [128, 128,   0],    #  5   tree
              [192, 128, 128],    #  6   sign symbol
              [ 64,  64, 128],    #  7   fence
              [ 64,   0, 128],    #  8   car
              [ 64,  64,   0],    #  9   pedestrian
              [  0, 128, 192],    # 10   bicyclist
              [  0,   0,   0]     # 11   don't care
        ]
    else:
        class_num = 20
        color_map = [
            #   R    G    B         ID   class
              [128,  64, 128],    #  0   road
              [244,  35, 232],    #  1   sidewalk
              [ 70,  70,  70],    #  2   building
              [102, 102, 156],    #  3   wall
              [190, 153, 153],    #  4   fence
              [153, 153, 153],    #  5   pole
              [250, 170,  30],    #  6   traffic light
              [220, 220,   0],    #  7   traffic sign
              [107, 142,  35],    #  8   vegetation
              [152, 251, 152],    #  9   terrain
              [ 70, 130, 180],    # 10   sky
              [220,  20,  60],    # 11   person
              [255,   0,   0],    # 12   rider
              [  0,   0, 142],    # 13   car
              [  0,   0,  70],    # 14   truck
              [  0,  60, 100],    # 15   bus
              [  0,  80, 100],    # 16   train
              [  0,   0, 230],    # 17   motorcycle
              [119,  11,  32],    # 18   bicycle
              [  0,   0,   0]     # 19   don't care
        ]
    
    #Create new array
    segmentationR = prediction.copy()
    segmentationG = prediction.copy()
    segmentationB = prediction.copy()
    
    #Start mapping
    if class_num>20:
      class_num = 20
    for i in xrange(class_num):
        segmentationR[segmentationR==i] = color_map[i][0]
        segmentationG[segmentationG==i] = color_map[i][1]
        segmentationB[segmentationB==i] = color_map[i][2]
    
    #Concatenate
    segmentationR = np.reshape(segmentationR, [height, width, 1])
    segmentationG = np.reshape(segmentationG, [height, width, 1])
    segmentationB = np.reshape(segmentationB, [height, width, 1])
    segmentation = np.concatenate((segmentationB, segmentationG, segmentationR), axis=2)
    
    return segmentation.astype(np.uint8)
    
def one_hot(annotations, batch_num, height=360, width=480, dataset="CamVid"):
    '''
    Transfer annotations to one hot form.
    
    INPUT:
    - annotations (Tensor): the original annotation with shape [batch, height, width, 1]
    - dataset (string): annotations belong to which dataset
    - batch_num (int): number of one batch
    
    OUTPUT:
    - annotations_one_hot (Tensor): one hot form of annotations with shape [batch, height, width, num_classes]
    '''
    
    if dataset=="CamVid":
        annotations_one_hot = tf.one_hot(annotations, 12, axis=-1)
        return annotations_one_hot
    elif dataset=="Cityscapes":
        annotations_one_hot_256 = tf.one_hot(annotations, 256, axis=-1)
        annotations_one_hot_19 = tf.slice(annotations_one_hot_256, [0, 0, 0, 0], [batch_num, height, width, 19])
        annotations_one_hot_255 = tf.slice(annotations_one_hot_256, [0, 0, 0, 254], [batch_num, height, width, 1])
        annotations_one_hot = tf.concat([annotations_one_hot_19, annotations_one_hot_255], axis=3)
        return annotations_one_hot
    logging.info("one hot transfer error")
    sys.exit(0)
    

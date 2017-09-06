from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf
import numpy as np
import sys

input_height = 360
input_width = 480

def get_slice_num(height=360, width=480):
    '''
    Calculate slice number.
    
    INPUTS:
    - height (int): the image original height
    - width (int): the image original width
    
    OUTPUTS:
    - slice_num (int): slice number
    '''
    rows = int(np.ceil(float(height)/float(input_height)))
    cols = int(np.ceil(float(width)/float(input_width)))
    slice_num = rows*cols
    return slice_num

def preprocess_ori(image, annotation=None, height=360, width=480):
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

    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    image.set_shape(shape=(height, width, 3))

    if not annotation == None:
        annotation = tf.image.resize_image_with_crop_or_pad(annotation, height, width)
        annotation.set_shape(shape=(height, width, 1))

        return image, annotation

    return image

def preprocess(image, annotation=None, height=360, width=480, filename=None):
    '''
    Performs preprocessing for one set of image and annotation for feeding into network.
    NO scaling of any sort will be done as per original paper.

    INPUTS:
    - image (Tensor): the image input 3D Tensor of shape [height, width, 3]
    - annotation (Tensor): the annotation input 3D Tensor of shape [height, width, 1]
    - height (int): the image original height
    - width (int): the image original width
    - filename (Tensor): option parameter, give if want to tie filename width image

    OUTPUTS:
    - images (Tensor): the reshaped image tensor of shape [slice_num, height, width, 3]
    - annotations (Tensor): the reshaped annotation tensor of shape [slice_num, height, width, 1]
    - filenames (Tensor): filename array of shape [slice_num, 1]
    - codes (Tensor): list of image code [slice_num, 1]
    '''

    #Convert the image and annotation dtypes to tf.float32 if needed
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.cast(image, tf.float32)

    #Slice parameters
    rows = int(np.ceil(float(height)/float(input_height)))
    cols = int(np.ceil(float(width)/float(input_width)))
    row_overlap = 0
    col_overlap = 0
    if rows>1:
        row_overlap = np.floor(float(input_height*rows-height)/float(rows-1))
    if cols>1:
        col_overlap = np.floor(float(input_width*cols-width)/float(cols-1))
    row_stride = input_height - row_overlap
    col_stride = input_width - col_overlap
    codes = np.zeros([int(rows*cols), 1])
    if filename!=None:
        filename = tf.reshape(filename, [1, 1])
	
    #Slicing image
    for row in xrange(rows):
        for col in xrange(cols):
            codes[int(row*cols+col)][0] = int(row*cols+col)
            if row+col==0:
                if filename!=None:
                    filenames = filename
                images = tf.slice(image, [0, 0, 0], [input_height, input_width, 3])
                images.set_shape(shape=(input_height, input_width, 3))
                images = tf.reshape(images, [1, input_height, input_width, 3])
            else:
              	aslice = tf.slice(image, [int(row*row_stride), int(col*col_stride), 0], [input_height, input_width, 3])
              	aslice.set_shape(shape=(input_height, input_width, 3))
              	aslice = tf.reshape(aslice, [1, input_height, input_width, 3])
              	images = tf.concat([images, aslice], 0)
                if filename!=None:
                    filenames = tf.concat([filenames, filename], 0)
    codes = tf.convert_to_tensor(codes)
    codes = tf.cast(codes, tf.int32)
	
    #Slicing annotation
    if annotation != None:
        for row in xrange(rows):
            for col in xrange(cols):
                if row+col==0:
                    annotations = tf.slice(annotation, [0, 0, 0], [input_height, input_width, 1])
                    annotations.set_shape(shape=(input_height, input_width, 1))
                    annotations = tf.reshape(annotations, [1, input_height, input_width, 1])
                else:
                    aslice = tf.slice(annotation, [int(row*row_stride), int(col*col_stride), 0], [input_height, input_width, 1])
                    aslice.set_shape(shape=(input_height, input_width, 1))
                    aslice = tf.reshape(aslice, [1, input_height, input_width, 1])
                    annotations = tf.concat([annotations, aslice], 0)
        if filename!=None:
            return images, annotations, filenames, codes
        else:
            return images, annotations, codes
        
    if filename!=None:
        return images, filenames, codes
    else:
        return images, codes

def overlap_random_choose(imageA, imageB):
    '''
    Choose either A or B as final result.
    
    INPUTS:
    - imageA, imageB (ndarray): two source image
    
    OUTPUTS:
    - result (ndarray): result of random choose from A or B with shape the same as inputs
    '''
    
    height, width = imageA.shape
    maskA = np.random.randint(2, size=(height, width))
    maskB = maskA + 1
    maskB[maskB==2] = 0
    result = imageA*maskA + imageB*maskB
    return result

def postprocess(images, height=360, width=480, rgb=False):
    '''
    Combine images back to origin size.
    
    INPUTS:
    - images (ndarray): slices of the prediction result with shape [batch, height, width], value: 0-class_num-1
    - height (int): the image original height
    - width (int): the image original width
    
    OUTPUTS:
    - image (ndarray): combined image with shape [height, width], value: 0-class_num-1
    '''
    
    #Slice parameters
    rows = int(np.ceil(float(height)/float(input_height)))
    cols = int(np.ceil(float(width)/float(input_width)))
    row_overlap = np.floor(float(input_height*rows-height)/float(rows-1))
    col_overlap = np.floor(float(input_width*cols-width)/float(cols-1))
    row_stride = input_height - row_overlap
    col_stride = input_width - col_overlap
    
    #Prepare images
    if rgb==True:
      image = np.zeros([height, width, 3])
    else:
      image = np.zeros([height, width])
    
    #Combining
    for row in xrange(rows):
        for col in xrange(cols):
            row_leftup = int(row*row_stride)
            col_leftup = int(col*col_stride)
            if rgb==True:
              image[row_leftup:row_leftup+input_height, col_leftup:col_leftup+input_width, :] = images[row*cols+col, :]
            else:
              image[row_leftup:row_leftup+input_height, col_leftup:col_leftup+input_width] = images[row*cols+col]
    if rgb==True:
      image = convert_to_opencv_form(image)
      
    return image
                

def produce_color_segmentation(prediction, height=360, width=480, dataset="CamVid"):
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
    elif dataset=="Cityscapes":
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
    elif dataset=="NYU":
        class_num = 5
        color_map = [
            #   R    G    B         ID   class
              [  0,   0,   0],    #  0   don't care
              [147, 161, 161],    #  1   structure
              [181, 137,   0],    #  2   prop
              [203,  75,  22],    #  3   furniture
              [  7,  54,  66],    #  4   floor
        ]
    else:
        logging.info("no specify dataset!")
        sys.exit(0)
    
    #Create new array
    segmentationR = prediction.copy()
    segmentationG = prediction.copy()
    segmentationB = prediction.copy()
    
    #Start mapping
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
    
def one_hot(annotations, batch_num, dataset="CamVid"):
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
        annotations_one_hot_19 = tf.slice(annotations_one_hot_256, [0, 0, 0, 0], [batch_num, input_height, input_width, 19])
        annotations_one_hot_255 = tf.slice(annotations_one_hot_256, [0, 0, 0, 254], [batch_num, input_height, input_width, 1])
        annotations_one_hot = tf.concat([annotations_one_hot_19, annotations_one_hot_255], axis=3)
        return annotations_one_hot
    elif dataset=="NYU":
        annotations_one_hot = tf.one_hot(annotations, 5, axis=-1)
        return annotations_one_hot
    logging.info("one hot transfer error")
    sys.exit(0)
    
def convert_to_opencv_form(image):
    '''
    Transfer image from RGB to BGR.
    
    INPUT:
    - image (ndarray): the original image with shape [height, width, 3], value: 0~1
    
    OUTPUT:
    - image (ndarray): transformed image with shape [height, width, 3], value: 0~255
    '''

    image *= 255
    image.astype(np.uint8)
    height, width, channel = image.shape
    R = np.reshape(image[:, :, 0], [height, width, 1])
    G = np.reshape(image[:, :, 1], [height, width, 1])
    B = np.reshape(image[:, :, 2], [height, width, 1])
    image = np.concatenate((B, G, R), axis=2)
    return image

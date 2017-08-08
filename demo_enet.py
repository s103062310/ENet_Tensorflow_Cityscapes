import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess, postprocess
import os, glob, time, cv2
import numpy as np
import matplotlib.pyplot as plt
slim = tf.contrib.slim

#============INPUT ARGUMENTS================
flags = tf.app.flags

#Directories
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('checkpoint_dir', './log/original', 'The checkpoint directory to restore your mode.l')
flags.DEFINE_string('logdir', './log/original_test', 'The log directory for event files created during test evaluation.')
flags.DEFINE_boolean('save_images', True, 'If True, saves 10 images to your logdir for visualization.')

#Evaluation information
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 10, 'The batch_size for evaluation.')
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 1, "The number of epochs to evaluate your model.")
flags.DEFINE_string('dataset', "CamVid", 'Which dataset to test')

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
image_height = FLAGS.image_height
image_width = FLAGS.image_width
num_epochs = FLAGS.num_epochs
save_images = FLAGS.save_images
print image_height, image_width

#Architectural changes
num_initial_blocks = FLAGS.num_initial_blocks
stage_two_repeat = FLAGS.stage_two_repeat
skip_connections = FLAGS.skip_connections

dataset_dir = FLAGS.dataset_dir
checkpoint_dir = FLAGS.checkpoint_dir
photo_dir = os.path.join(FLAGS.logdir, "images")
logdir = FLAGS.logdir
dataset = FLAGS.dataset

#===============PREPARATION FOR TRAINING==================
#Checkpoint directories
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

#Dataset directories
if dataset=="CamVid":
  image_files = os.path.join(dataset_dir, "val", "*.png")
elif dataset=="Cityscapes":
  image_files = os.path.join(dataset_dir, "demo", "*", "*.png")
  print image_files
image_files = glob.glob(image_files)
image_files.sort()
print image_files

num_batches_per_epoch = len(image_files) / batch_size
num_steps_per_epoch = num_batches_per_epoch

#=============EVALUATION=================
def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        #===================TEST BRANCH=======================
        #Load the files into one input queue
        images = tf.convert_to_tensor(image_files)
        input_queue = tf.train.slice_input_producer([images])
        
        #Decode the image and annotation raw content
        filename = input_queue[0]
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_image(image, channels=3)
        
        #preprocess and batch up the image and annotation
        preprocessed_image = preprocess(image, None, image_height, image_width)
        images, filenames = tf.train.batch([preprocessed_image, filename], batch_size=batch_size, allow_smaller_final_batch=True)
        
        #Create the model inference
        with slim.arg_scope(ENet_arg_scope()):
            logits, probabilities = ENet(images,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None,
                                         num_initial_blocks=num_initial_blocks,
                                         stage_two_repeat=stage_two_repeat,
                                         skip_connections=skip_connections)

        # Set up the variables to restore and restoring function from a saver.
        exclude = []
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = logdir, summary_op = None, init_fn=restore_fn)

        #Run the managed session
        with sv.managed_session() as sess:
        
            #Save the images
            if save_images:
                if not os.path.exists(photo_dir):
                    os.mkdir(photo_dir)

                #Segmentation
                logging.info('Total Steps: %d', int(num_steps_per_epoch * num_epochs))
                logging.info('Saving the images now...')
                for step in range(int(num_steps_per_epoch * num_epochs)):
                    start_time = time.time()
                    predictions_val, filename_val = sess.run([predictions, filenames])
                    time_elapsed = time.time() - start_time
                    logging.info('step %d  %.2f(sec/step)  %.2f (fps)', step, time_elapsed, batch_size/time_elapsed)
                    
                    #Save the image visualizations
                    for i in xrange(batch_size):
                        filename_val_single = filename_val[i].split('/')
                        filename_val_single = filename_val_single[len(filename_val_single)-1]
                        segmentation = postprocess(predictions_val[i], image_height, image_width, dataset)
                        cv2.imwrite(photo_dir+"/image_" + filename_val_single, segmentation)

if __name__ == '__main__':
    run()
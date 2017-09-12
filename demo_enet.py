import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess_ori, postprocess, produce_color_segmentation
import os, glob, time, cv2
import numpy as np
slim = tf.contrib.slim

#============INPUT ARGUMENTS================
flags = tf.app.flags

#Directories
flags.DEFINE_string('checkpoint_dir', '../log/original', 'The checkpoint directory to restore your mode.l')
flags.DEFINE_string('photo_dir', '../log/original_test', 'The log directory for event files created during test evaluation.')
flags.DEFINE_boolean('save_images', True, 'If True, saves 10 images to your for visualization.')
flags.DEFINE_string('dataset', "CamVid", 'Which dataset to test')

#Evaluation information
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('batch_size', 5, 'The batch_size for testing.')

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
image_height = FLAGS.image_height
image_width = FLAGS.image_width
save_images = FLAGS.save_images
batch_size = FLAGS.batch_size

#Architectural changes
num_initial_blocks = FLAGS.num_initial_blocks
stage_two_repeat = FLAGS.stage_two_repeat
skip_connections = FLAGS.skip_connections

photo_dir = FLAGS.photo_dir
checkpoint_dir = FLAGS.checkpoint_dir
dataset = FLAGS.dataset

#===============PREPARATION FOR TRAINING==================
#Checkpoint directories
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

#Dataset directories
if dataset=="CamVid":
  dataset_dir = "../dataset"	#Change dataset location => modify here
  image_files = os.path.join(dataset_dir, "val", "*.png")
elif dataset=="Cityscapes":
  dataset_dir = "../cityscapes"	#Change dataset location => modify here
  image_files = os.path.join(dataset_dir, "demo", "*", "*.png")
elif dataset=="NYU":
  dataset_dir = "../NYU" #Change dataset location => modify here
  image_files = os.path.join(dataset_dir, "testing", "*", "*_colors.png")
else:
  image_files = os.path.join("../cityscapes", "leftImg8bit",  "val", "frankfurt", "frankfurt_000000_000294_leftImg8bit.png")

image_files = glob.glob(image_files)
image_files.sort()

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
        preprocessed_image = preprocess_ori(image, None, image_height, image_width)
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

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = photo_dir, summary_op = None, init_fn=restore_fn)

        #Run the managed session
        with sv.managed_session() as sess:
        
            #Save the images
            if save_images:
                if not os.path.exists(photo_dir):
                    os.mkdir(photo_dir)

                #Segmentation
                total_time = 0
                logging.info('Total Steps: %d', int(num_steps_per_epoch))
                logging.info('Saving the images now...')
                for step in range(int(num_steps_per_epoch)):
                    start_time = time.time()
                    predictions_val, filename_val = sess.run([predictions, filenames])
                    time_elapsed = time.time() - start_time
                    logging.info('step %d  %.2f(sec/step)  %.2f (fps)', step, time_elapsed, batch_size/time_elapsed)
                    if step!=0:
                        total_time = total_time + time_elapsed
                    
                    for i in xrange(batch_size):
                        segmentation = produce_color_segmentation(predictions_val[i], image_height, image_width, dataset)
                        filename = filename_val[i].split('/')
                        filename = filename[len(filename)-1]
                        filename = photo_dir+"/trainResult_" + filename
                        cv2.imwrite(filename, segmentation)
                        
                logging.info('Average speed: %.2f fps', len(image_files)/total_time)

if __name__ == '__main__':
    run()
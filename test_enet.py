import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess, postprocess, one_hot, input_height, input_width, get_slice_num, produce_color_segmentation
import os, glob, cv2, time
import numpy as np
slim = tf.contrib.slim

#============INPUT ARGUMENTS================
flags = tf.app.flags

#Directories
flags.DEFINE_string('checkpoint_dir', '../log/original', 'The checkpoint directory to restore your mode.l')
flags.DEFINE_string('logdir', '../log/original_test', 'The log directory for event files created during test evaluation.')
flags.DEFINE_boolean('save_images', True, 'If True, saves 10 images to your logdir for visualization.')
flags.DEFINE_string('dataset', "CamVid", 'Which dataset to train')

#Evaluation information
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 1, "The number of epochs to evaluate your model.")

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
image_height = FLAGS.image_height
image_width = FLAGS.image_width
num_epochs = FLAGS.num_epochs
batch_size = get_slice_num(image_height, image_width)

save_images = FLAGS.save_images
dataset = FLAGS.dataset

#Architectural changes
num_initial_blocks = FLAGS.num_initial_blocks
stage_two_repeat = FLAGS.stage_two_repeat
skip_connections = FLAGS.skip_connections

checkpoint_dir = FLAGS.checkpoint_dir
photo_dir = os.path.join(FLAGS.logdir, "images")
logdir = FLAGS.logdir

#===============PREPARATION FOR TRAINING==================
#Checkpoint directories
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

#Dataset directories
#CamVid
if dataset=="CamVid":
    dataset_dir = "../dataset"	#Change dataset location => modify here
    image_files = os.path.join(dataset_dir, "test", "*.png")
    annotation_files = os.path.join(dataset_dir, "testannot", "*.png")

#Cityscapes
elif dataset=="Cityscapes":
    dataset_dir = "../cityscapes"	#Change dataset location => modify here
    image_files = os.path.join(dataset_dir, "leftImg8bit", "val", "frankfurt", "*_leftImg8bit.png")
    annotation_files = os.path.join(dataset_dir, "gtFine", "val", "frankfurt", "*_gtFine_labelTrainIds.png")

image_files = glob.glob(image_files)
image_files.sort()
annotation_files = glob.glob(annotation_files)
annotation_files.sort()

num_batches_per_epoch = len(image_files)
num_steps_per_epoch = num_batches_per_epoch

#=============EVALUATION=================
def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        #===================TEST BRANCH=======================
        #Load the files into one input queue
        images = tf.convert_to_tensor(image_files)
        annotations = tf.convert_to_tensor(annotation_files)
        input_queue = tf.train.slice_input_producer([images, annotations], shuffle=False)
        
        #Decode the image and annotation raw content
        filename = input_queue[0]
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_image(image, channels=3)
        annotation = tf.read_file(input_queue[1])
        annotation = tf.image.decode_image(annotation)
        
        #preprocess and batch up the image and annotation
        preprocessed_images, preprocessed_annotations, correspond_filenames, image_codes = preprocess(image, annotation, image_height, image_width, filename)
        process_queue = tf.train.slice_input_producer([preprocessed_images, preprocessed_annotations, correspond_filenames, image_codes], shuffle=False)
        image = process_queue[0]
        annotation = process_queue[1]
        filename = process_queue[2]
        code = process_queue[3]
        images, annotations, filenames, codes = tf.train.batch([image, annotation, filename, code], batch_size=batch_size, allow_smaller_final_batch=True)
        
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
        
        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations = tf.reshape(annotations, shape=[batch_size, input_height, input_width])
        annotations_ohe = one_hot(annotations, batch_size, dataset)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, annotations)
        mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions, labels=annotations, num_classes=num_classes)
        per_class_accuracy, per_class_accuracy_update = tf.metrics.mean_per_class_accuracy(labels=annotations, predictions=predictions, num_classes=num_classes)
        metrics_op = tf.group(accuracy_update, mean_IOU_update, per_class_accuracy_update)

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step

        #Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value, mean_IOU_value, per_class_accuracy_value = sess.run([metrics_op, global_step_op, accuracy, mean_IOU, per_class_accuracy])
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f     Streaming Mean IOU: %.4f     Per-class Accuracy: %.4f (%.2f sec/step)',
                         global_step_count, accuracy_value, mean_IOU_value, per_class_accuracy_value, time_elapsed)

            return accuracy_value, mean_IOU_value, per_class_accuracy_value

        #Create your summaries
        tf.summary.scalar('Monitor/test_accuracy', accuracy)
        tf.summary.scalar('Monitor/test_mean_per_class_accuracy', per_class_accuracy)
        tf.summary.scalar('Monitor/test_mean_IOU', mean_IOU)
        my_summary_op = tf.summary.merge_all()

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = logdir, summary_op = None, init_fn=restore_fn)

        #Run the managed session
        with sv.managed_session() as sess:
            
            '''for step in range(int(num_steps_per_epoch * num_epochs)):
                #print vital information every start of the epoch as always
                if step % num_batches_per_epoch == 0:
                    accuracy_value, mean_IOU_value = sess.run([accuracy, mean_IOU])
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f', accuracy_value)
                    logging.info('Current Streaming Mean IOU: %.4f', mean_IOU_value)
                    
                #Compute summaries every 10 steps and continue evaluating
                if step % 10 == 0:
                    test_accuracy, test_mean_IOU, test_per_class_accuracy = eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    
                #Otherwise just run as per normal
                else:
                    test_accuracy, test_mean_IOU, test_per_class_accuracy = eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)

            #At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', test_accuracy)
            logging.info('Final Mean IOU: %.4f', test_mean_IOU)
            logging.info('Final Per Class Accuracy %.4f', test_per_class_accuracy)

            #Show end of evaluation
            logging.info('Finished evaluating!')'''

            #Save the images
            if save_images:
                if not os.path.exists(photo_dir):
                    os.mkdir(photo_dir)
                
                #Segmentation
                logging.info('Total Steps: %d', int(num_steps_per_epoch * num_epochs))
                logging.info('Saving the images now...')
                for step in range(int(num_steps_per_epoch * num_epochs)):
                    start_time = time.time()
                    predictions_val, filename_val, image_val = sess.run([predictions, filenames, images])
                    time_elapsed = time.time() - start_time
                    logging.info('step %d  %.2f(sec/step)  %.2f (fps-360*480) %.2f (fps-1024*2048)', step, time_elapsed, batch_size/time_elapsed, 1/time_elapsed)
                    
                    combine = postprocess(predictions_val, image_height, image_width)
                    segmentation = produce_color_segmentation(combine, image_height, image_width, dataset)
                    filename = filename_val[0][0].split('/')
                    filename = filename[len(filename)-1]
                    filename1 = photo_dir+"/result_" + filename
                    filename2 = photo_dir+"/image_" + filename
                    cv2.imwrite(filename1, segmentation)
                    image_combine = postprocess(image_val, image_height, image_width, True)
                    cv2.imwrite(filename2, image_combine)
                    
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

if __name__ == '__main__':
    run()
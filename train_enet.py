import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess_ori, one_hot, produce_color_segmentation
from get_class_weights import ENet_weighing, median_frequency_balancing
import os, glob, time, cv2
import numpy as np
slim = tf.contrib.slim

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Directory arguments
flags.DEFINE_string('logdir', '../log/original', 'The log directory to save your checkpoint and event files.')
flags.DEFINE_boolean('save_images', False, 'Whether or not to save your images.')
flags.DEFINE_boolean('combine_dataset', False, 'If True, combines the validation with the train dataset.')
flags.DEFINE_string('dataset', "CamVid", 'Which dataset to train')

#Training arguments
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 5, 'The batch_size for training.')
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('eval_batch_size', 5, 'The batch_size for validation.')
flags.DEFINE_integer('num_epochs', 300, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 30, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 1e-1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 5e-4, 'The initial learning rate for your training.')
flags.DEFINE_string('weighting', "MFB", 'Choice of Median Frequency Balancing or the custom ENet class weights.')

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
eval_batch_size = FLAGS.eval_batch_size
combine_dataset = FLAGS.combine_dataset
dataset = FLAGS.dataset

#Training parameters
initial_learning_rate = FLAGS.initial_learning_rate
num_epochs_before_decay = FLAGS.num_epochs_before_decay
num_epochs =FLAGS.num_epochs
learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
weight_decay = FLAGS.weight_decay
epsilon = 1e-8

#Architectural changes
num_initial_blocks = FLAGS.num_initial_blocks
stage_two_repeat = FLAGS.stage_two_repeat
skip_connections = FLAGS.skip_connections

#Use median frequency balancing or not
weighting = FLAGS.weighting

#Visualization and where to save images
save_images = FLAGS.save_images
photo_dir = os.path.join(FLAGS.logdir, "images")

#Directories
logdir = FLAGS.logdir

#===============PREPARATION FOR TRAINING==================
#Get the images into a list

#CamVid
if dataset=="CamVid":
    dataset_dir = "../dataset"	#Change dataset location => modify here
    image_files = os.path.join(dataset_dir, "train", "*.png")
    annotation_files = os.path.join(dataset_dir, "trainannot", "*.png")
    image_val_files = os.path.join(dataset_dir, "val", "*.png")
    annotation_val_files = os.path.join(dataset_dir, "valannot", "*.png")

#Cityscapes
elif dataset=="Cityscapes":
    dataset_dir = "../cityscapes"	#Change dataset location => modify here
    image_files = os.path.join(dataset_dir, "leftImg8bit", "train", "*", "*_leftImg8bit.png")
    annotation_files = os.path.join(dataset_dir, "gtFine", "train", "*", "*_gtFine_labelTrainIds.png")
    image_val_files = os.path.join(dataset_dir, "leftImg8bit", "val", "*", "*_leftImg8bit.png")
    annotation_val_files = os.path.join(dataset_dir, "gtFine", "val", "*", "*_gtFine_labelTrainIds.png")

#NYU
elif dataset=="NYU":
    dataset_dir = "../NYU"	#Change dataset location => modify here
    image_files = os.path.join(dataset_dir, "training", "*", "*_colors.png")
    annotation_files = os.path.join(dataset_dir, "training", "*", "*_ground_truth_id.png")
    image_val_files = os.path.join(dataset_dir, "testing", "*", "*_colors.png")
    annotation_val_files = os.path.join(dataset_dir, "testing", "*", "*_ground_truth_id.png")

elif dataset=="ADE":
    dataset_dir = "../ADE" #Change dataset location => modify here
    image_files = os.path.join(dataset_dir, "images", "training", "ADE_train_*.jpg")
    annotation_files = os.path.join(dataset_dir, "annotations_relabel", "training", "ADE_train_*.png")
    image_val_files = os.path.join(dataset_dir, "images", "validation", "ADE_val_*.jpg")
    annotation_val_files = os.path.join(dataset_dir, "annotations_relabel", "validation", "ADE_val_*.png")

#Get complete file path
image_files = glob.glob(image_files)
annotation_files = glob.glob(annotation_files)
image_val_files = glob.glob(image_val_files)
annotation_val_files = glob.glob(annotation_val_files)
image_files.sort()
annotation_files.sort()
image_val_files.sort()
annotation_val_files.sort()

# Combine dataset
if combine_dataset:
    image_files += image_val_files
    annotation_files += annotation_val_files

#Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = len(image_files) / batch_size
num_steps_per_epoch = num_batches_per_epoch
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

#=================CLASS WEIGHTS===============================
#Median frequency balancing class_weights
print "getting weight..."
if weighting == "MFB":
    class_weights = median_frequency_balancing()
    print "========= Median Frequency Balancing Class Weights =========\n", class_weights

#Inverse weighing probability class weights
elif weighting == "ENET":
    if dataset=="Cityscapes":
        class_weights = [3.0446304066119358, 12.859237726097096, 4.5106188144793764, 38.149414675592844, 35.240414651744551, 31.480455482676099, 45.777966082085818, 39.669614104133458, 6.0674434654058942, 32.15635357347918, 17.129877639929358, 31.579296636222271, 47.339784184712087, 11.602562167110021, 44.596951840626446, 45.23276260758076, 45.280616737115544, 48.148015881262801, 41.924773811374628, 0.0]
    elif dataset=="ADE":
        class_weights = [0.0, 5.4675178830470745, 7.4601094583312433, 8.7120767862064774, 11.465808348047995, 13.712429551972468, 14.413321273760605, 15.705530598098903, 22.080140191392989, 23.803016496287306, 24.985279926263889, 25.024199990546769, 26.016846374411703, 26.636528875446555, 27.368116740842485, 30.256841081755546, 31.104755166055948, 31.284902245688205, 32.136032211394443, 31.796023219341777, 31.929518751022876, 32.321876575080822, 35.631894924140624, 36.450061812083817, 36.754749501017052, 37.397204524932938, 37.600648277390256]
    else:
        class_weights = ENet_weighing(annotation_files, num_classes)
    print "========= ENet Class Weights =========\n", class_weights

#============= TRAINING =================
def weighted_cross_entropy(onehot_labels, logits, class_weights):
    '''
    A quick wrapper to compute weighted cross entropy. 

    ------------------
    Technical Details
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want. 
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.

    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------

    INPUTS:
    - onehot_labels(Tensor): the one-hot encoded labels of shape (batch_size, height, width, num_classes)
    - logits(Tensor): the logits output from the model that is of shape (batch_size, height, width, num_classes)
    - class_weights(list): A list where each index is the class label and the value of the index is the class weight.

    OUTPUTS:
    - loss(Tensor): a scalar Tensor that is the weighted cross entropy loss output.
    '''
    weights = onehot_labels * class_weights
    weights = tf.reduce_sum(weights, 3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)

    return loss

def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        #===================TRAINING BRANCH=======================
        #Load the files into one input queue
        images = tf.convert_to_tensor(image_files)
        annotations = tf.convert_to_tensor(annotation_files)
        input_queue = tf.train.slice_input_producer([images, annotations]) #Slice_input producer shuffles the data by default.

        #Decode the image and annotation raw content
        filename = input_queue[0]
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_image(image, channels=3)
        annotation = tf.read_file(input_queue[1])
        annotation = tf.image.decode_image(annotation)

        #preprocess the images and annotations
        preprocessed_image, preprocessed_annotation = preprocess_ori(image, annotation, image_height, image_width)
        
        #Batch up the images and annotations
        images, annotations = tf.train.batch([preprocessed_image, preprocessed_annotation], batch_size=batch_size, allow_smaller_final_batch=True)

        #Create the model inference
        with slim.arg_scope(ENet_arg_scope(weight_decay=weight_decay)):
            logits, probabilities = ENet(images,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None,
                                         num_initial_blocks=num_initial_blocks,
                                         stage_two_repeat=stage_two_repeat,
                                         skip_connections=skip_connections)
        
        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations = tf.reshape(annotations, shape=[batch_size, image_height, image_width])
        annotations_ohe = one_hot(annotations, batch_size, dataset)
        
        #Actually compute the loss
        loss = weighted_cross_entropy(logits=logits, onehot_labels=annotations_ohe, class_weights=class_weights)
        total_loss = tf.losses.get_total_loss()

        #Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        #Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

        #Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)

        #Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, annotations)
        mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions, labels=annotations, num_classes=num_classes)
        metrics_op = tf.group(accuracy_update, mean_IOU_update)

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, metrics_op):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, accuracy_val, mean_IOU_val, _ = sess.run([train_op, global_step, accuracy, mean_IOU, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to show some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)(%.2f fps)    Current Streaming Accuracy: %.4f    Current Mean IOU: %.4f', global_step_count, total_loss, time_elapsed/batch_size, batch_size/time_elapsed, accuracy_val, mean_IOU_val)

            return total_loss, accuracy_val, mean_IOU_val

        #================VALIDATION BRANCH========================
        #Load the files into one input queue
        images_val = tf.convert_to_tensor(image_val_files)
        annotations_val = tf.convert_to_tensor(annotation_val_files)
        input_queue_val = tf.train.slice_input_producer([images_val, annotations_val])

        #Decode the image and annotation raw content
        filename_val = input_queue_val[0]
        image_val = tf.read_file(input_queue_val[0])
        image_val = tf.image.decode_jpeg(image_val, channels=3)
        annotation_val = tf.read_file(input_queue_val[1])
        annotation_val = tf.image.decode_png(annotation_val)

        #preprocess the images and annotations
        preprocessed_image_val, preprocessed_annotation_val = preprocess_ori(image_val, annotation_val, image_height, image_width)
        
        #Batch up the image and annotation
        images_val, annotations_val, filenames_val = tf.train.batch([preprocessed_image_val, preprocessed_annotation_val, filename_val], batch_size=eval_batch_size, allow_smaller_final_batch=True)

        with slim.arg_scope(ENet_arg_scope(weight_decay=weight_decay)):
            logits_val, probabilities_val = ENet(images_val,
                                                 num_classes,
                                                 batch_size=eval_batch_size,
                                                 is_training=True,
                                                 reuse=True,
                                                 num_initial_blocks=num_initial_blocks,
                                                 stage_two_repeat=stage_two_repeat,
                                                 skip_connections=skip_connections)

        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations_val = tf.reshape(annotations_val, shape=[eval_batch_size, image_height, image_width])
        annotations_ohe_val = one_hot(annotations_val, batch_size, dataset)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded. ----> Should we use OHE instead?
        predictions_val = tf.argmax(probabilities_val, -1)
        accuracy_val, accuracy_val_update = tf.contrib.metrics.streaming_accuracy(predictions_val, annotations_val)
        mean_IOU_val, mean_IOU_val_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions_val, labels=annotations_val, num_classes=num_classes)
        metrics_op_val = tf.group(accuracy_val_update, mean_IOU_val_update)

        #Create an output for showing the segmentation output of validation images
        segmentation_output_val = tf.cast(predictions_val, dtype=tf.float32)
        segmentation_output_val = tf.reshape(segmentation_output_val, shape=[-1, image_height, image_width, 1])
        segmentation_ground_truth_val = tf.cast(annotations_val, dtype=tf.float32)
        segmentation_ground_truth_val = tf.reshape(segmentation_ground_truth_val, shape=[-1, image_height, image_width, 1])

        def eval_step(sess, metrics_op):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, accuracy_value, mean_IOU_value = sess.run([metrics_op, accuracy_val, mean_IOU_val])
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('---VALIDATION--- Validation Accuracy: %.4f    Validation Mean IOU: %.4f    (%.2f sec/step)(%.2f fps)', accuracy_value, mean_IOU_value, time_elapsed/eval_batch_size, eval_batch_size/time_elapsed)

            return accuracy_value, mean_IOU_value

        #=====================================================

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('Monitor/Total_Loss', total_loss)
        tf.summary.scalar('Monitor/validation_accuracy', accuracy_val)
        tf.summary.scalar('Monitor/training_accuracy', accuracy)
        tf.summary.scalar('Monitor/validation_mean_IOU', mean_IOU_val)
        tf.summary.scalar('Monitor/training_mean_IOU', mean_IOU)
        tf.summary.scalar('Monitor/learning_rate', lr)
        tf.summary.image('Images/Validation_original_image', images_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_output', segmentation_output_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_ground_truth', segmentation_ground_truth_val, max_outputs=1)
        my_summary_op = tf.summary.merge_all()
        
        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=logdir, summary_op=None, init_fn=None)

        # Run the managed session
        with sv.managed_session() as sess:
        
            for step in xrange(int(num_steps_per_epoch * num_epochs)):
                #At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value = sess.run([lr])
                    logging.info('Current Learning Rate: %s', learning_rate_value)

                #Log the summaries every 10 steps or every end of epoch, which ever lower.
                if step % min(num_steps_per_epoch, 10) == 0:
                    loss, training_accuracy, training_mean_IOU = train_step(sess, train_op, sv.global_step, metrics_op=metrics_op)

                    #Check the validation data only at every third of an epoch
                    num_to_val = num_steps_per_epoch / 3
                    if step % num_to_val == 0:
                        for i in xrange(len(image_val_files) / eval_batch_size):
                            validation_accuracy, validation_mean_IOU = eval_step(sess, metrics_op_val)

                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    
                #If not, simply run the training step
                else:
                    loss, training_accuracy,training_mean_IOU = train_step(sess, train_op, sv.global_step, metrics_op=metrics_op)

            #We log the final training loss
            logging.info('Final Loss: %s', loss)
            logging.info('Final Training Accuracy: %s', training_accuracy)
            logging.info('Final Training Mean IOU: %s', training_mean_IOU)
            logging.info('Final Validation Accuracy: %s', validation_accuracy)
            logging.info('Final Validation Mean IOU: %s', validation_mean_IOU)

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

            if save_images:
                if not os.path.exists(photo_dir):
                    os.mkdir(photo_dir)
                
                #Plot the predictions - check validation images only
                logging.info('Total Steps: %d', len(image_val_files) / eval_batch_size)
                logging.info('Saving the images now...')
                for step in xrange(len(image_val_files) / eval_batch_size):
                    start_time = time.time()
                    predictions_value, filenames_value = sess.run([predictions_val, filenames_val])
                    time_elapsed = time.time() - start_time
                    logging.info('step %d  %.2f(sec/step)  %.2f (fps)', step, time_elapsed/eval_batch_size, eval_batch_size/time_elapsed)
                    
                    for i in xrange(eval_batch_size):
                        segmentation = produce_color_segmentation(predictions_value[i], image_height, image_width, dataset)
                        filename = filenames_value[i].split('/')
                        filename = filename[len(filename)-1]
                        filename = photo_dir+"/trainResult_" + filename
                        cv2.imwrite(filename, segmentation)
            
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

if __name__ == '__main__':
    run()

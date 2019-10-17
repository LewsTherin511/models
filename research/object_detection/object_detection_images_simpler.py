###################
###   Imports   ###
###################

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops

# imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util



#---------------------------#
#     Model preparation     #
#---------------------------#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by
# changing `PATH_TO_CKPT` to point to a new .pb file. By default we use an "SSD with Mobilenet" model here. See the [
# detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md
# ) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


#################################################################
#################################################################
#################################################################
# # Setting model name
# MODEL_NAME = 'models_ZOO/ssd_mobilenet_v2_oid_v4_2018_12_12'
# # MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#
# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
#
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'oid_v4_label_map.pbtxt')
#
# NUM_CLASSES = 601
#################################################################
#################################################################
#################################################################
# What model to download.
MODEL_NAME = 'custom_inference_graph/900_OI/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'guitar_label_map.pbtxt')
NUM_CLASSES = 1
#################################################################
#################################################################
#################################################################






# # Download Model if needed, and extract
# if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
# 	print ('Downloading the model')
# 	opener = urllib.request.URLopener()
# 	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# 	tar_file = tarfile.open(MODEL_FILE)
# 	for file in tar_file.getmembers():
# 		file_name = os.path.basename(file.name)
# 		if 'frozen_inference_graph.pb' in file_name:
# 			tar_file.extract(file, os.getcwd())
# 	print ('Download complete')
# else:
# 	print ('Model already exists')



# ## Load a (frozen) Tensorflow model into memory.
# normal graph used for detection
detection_graph = tf.Graph()
with detection_graph.as_default():
	# serialized graph to read frozen model from .pb file
	graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as f:
		graph_def.ParseFromString(f.read())
		# import serialized graph to 'detection_graph' (set as default above)
		tf.import_graph_def(graph_def, name='')




# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`. Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)





# ## Helper code
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	channel_dict = {'L': 1, 'RGB': 3}  # 'L' for Grayscale, 'RGB' : for 3 channel images
	return np.array(image.getdata()).reshape((im_height, im_width, channel_dict[image.mode])).astype(np.uint8)



#-----------------------#
#   Detection routine   #
#-----------------------#
def run_inference_for_single_image(image_np_expanded, detection_graph):
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# tensor in the graph corresponding to the picture to analyze, must be fed with picture during Session
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represents level of confidence for each of object, score is shown on the result image together with class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			# Actual detection.
			(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
																feed_dict={image_tensor: image_np_expanded})
			return (boxes, scores, classes, num_detections)
#-----------------------#
#-----------------------#



#-----------------------#
#   grayscale to RGB    #
#-----------------------#
def Check3D(image_np):
	if (image_np.shape[2] != 3):
		image_np = np.broadcast_to(image_np, (image_np.shape[0], image_np.shape[1], 3)).copy()  # Duplicating the Content
	return image_np



# #------------------------------#
# #   Send IMAGES to detection   #
# #------------------------------#
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
test_img_path = "test_images/"

# test_images_list = [image_name for image_name in os.listdir(test_img_path)]
# num_test_images = len(test_images_list)

for image_name in os.listdir(test_img_path):
	image_full_path = test_img_path + image_name
	image = Image.open(image_full_path)
	# array-based representation of image will be used later to prepare result image with boxes and labels on it
	image_np = load_image_into_numpy_array(image)

	# check if image is RGB or grayscale, and if grayscale convert to RGB
	image_np = Check3D(image_np)

	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)

	# Actual detection.
	(boxes, scores, classes, num_detections) = run_inference_for_single_image(image_np_expanded, detection_graph)

	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(image_np,
													   np.squeeze(boxes),
													   np.squeeze(classes).astype(np.int32),
													   np.squeeze(scores),
													   category_index,
													   use_normalized_coordinates=True,
													   line_thickness=8)

	plt.figure(figsize=IMAGE_SIZE)
	plt.imshow(image_np)
	plt.imsave(f'./test_images_results/900_oi/{image_name}', image_np)
	plt.close()
# #------------------------------#
# #------------------------------#
# #------------------------------#




#--------------------------------------------------------------------------#
#                          Send WEBCAM to detection                        #
#                              INSANELY SLOW!!!                            #
#           for webcam, can't send frame by frame to detection             #
#   it's better to take frames during tf.sess and analyze them real-time   #
#--------------------------------------------------------------------------#
# # intializing the web camera device
# cap = cv2.VideoCapture(0)
# ret = True
# while (ret):
# 	# take frame from webcam
# 	ret,image_np = cap.read()
# 	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
# 	image_np_expanded = np.expand_dims(image_np, axis=0)
# 	# Send current frame to actual detection
# 	(boxes, scores, classes, num_detections) = run_inference_for_single_image(image_np_expanded, detection_graph)
# 	# Visualization of the results of a detection.
# 	vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
# 												   np.squeeze(classes).astype(np.int32),
# 												   np.squeeze(scores),
# 												   category_index, use_normalized_coordinates=True,
# 												   line_thickness=8)
#
# 	cv2.imshow('image', cv2.resize(image_np, (1280, 960)))
#
# 	if cv2.waitKey(25) & 0xFF == ord('q'):
# 		cv2.destroyAllWindows()
# 		cap.release()
# 		break
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

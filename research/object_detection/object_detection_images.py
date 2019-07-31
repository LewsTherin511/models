# # Object Detection Demo


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

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util




from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
	raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')





#---------------------------#
#     Model preparation     #
#---------------------------#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by
# changing `PATH_TO_CKPT` to point to a new .pb file. By default we use an "SSD with Mobilenet" model here. See the [
# detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md
# ) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_complete_label_map.pbtxt')
# PATH_TO_LABELS = os.path.join('data', 'oid_v4_label_map.pbtxt')


# Download Model if needed, and extract
if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
		file_name = os.path.basename(file.name)
		if 'frozen_inference_graph.pb' in file_name:
			tar_file.extract(file, os.getcwd())
	print ('Download complete')
else:
	print ('Model already exists')



# ## Load a (frozen) Tensorflow model into memory.
# normal graph used for detection
detection_graph = tf.Graph()
with detection_graph.as_default():
	# serialized graph to read frozen model from .pb file
	graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as f:
		graph_def.ParseFromString(f.read())
		# import serialized graph to 'detection_graph' (set as default above)
		tf.import_graph_def(graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`. Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



# ## Helper code
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)



#-----------------------#
#   Detection routine   #
#-----------------------#
def run_inference_for_single_image(image_np_expanded, graph):
	with graph.as_default():
		with tf.Session() as sess:
			# Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			# names of all tensors present in the graph
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}

			# check if any of the tensors in the graph is named as one of the keys below
			for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
				tensor_name = key + ':0'
				# if any tensor has same name as one of the keys, add tensor to 'tensor_dict' using that key as label
				# Es. (tensor_dict['detection_classes'] = detection_classes)
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

			# --------------------------------------------------------------------------------------
			# if 'detection mask' is one of tensors found in graph, do SOMETHING WITH IT???
			# --------------------------------------------------------------------------------------
			if 'detection_masks' in tensor_dict:
				# The following processing is only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					detection_masks, detection_boxes, image_np_expanded.shape[1], image_np_expanded.shape[2])
				detection_masks_reframed = tf.cast(
					tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
			# --------------------------------------------------------------------------------------

			# tensor in the graph corresponding to the picture to analyze, must be fed with picture during Session
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Actual detection (same as in simplified version, just instead of listing the tensors
			# to evaluate you give 'tensor_dict' as argument)
			output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict
#-----------------------#
#-----------------------#




# #------------------------------#
# #   Send IMAGES to detection   #
# #------------------------------#
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
test_img_path = "test_images/"
num_test_images = len([f for f in os.listdir(test_img_path) if os.path.isfile(os.path.join(test_img_path, f))])
test_img_list = [os.path.join(test_img_path, f'image{i:02d}.jpg') for i in range(1, num_test_images+1) ]
img_count = 0
for image_path in test_img_list:
	image = Image.open(image_path)
	# array-based representation of image will be used later to prepare result image with boxes and labels on it
	image_np = load_image_into_numpy_array(image)
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)


	# Actual detection.
	output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(image_np,
													   output_dict['detection_boxes'],
													   output_dict['detection_classes'],
													   output_dict['detection_scores'],
													   category_index,
													   instance_masks=output_dict.get('detection_masks'),
													   use_normalized_coordinates=True,
													   line_thickness=8)
	plt.figure(figsize=IMAGE_SIZE)
	plt.imshow(image_np)
	plt.imsave(f'./results/img_{img_count:02d}.jpg', image_np)
	img_count+=1
# #------------------------------#
# #------------------------------#
# #------------------------------#
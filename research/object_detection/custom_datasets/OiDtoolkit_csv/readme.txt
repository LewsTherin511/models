
****************************************************************************************************************
---   Creating train_labels.txt and test_labels.txt starting from images and labels provided by OiD script   ---
---      Determine % of train/test images, resizes them and moves them to the train and test folders         ---
****************************************************************************************************************


Requirements:
input_images -> all images from one class from OiD (it should work for more classes too, have to check better)
input_txts -> txt files from OiD

arrange_data.py:
	* specify max_pixels = number of pixels along the longer size of the image
	* if longer size > max_pixels, reduce it to max_pixels and resizes other side accordingly
	* insert test_percentage -> percentage of images to use in test subset
	* program moves images to 'output_images_train' and 'output_images_test' folders, and creates 'test_labels.csv' and 'train_labels.csv' in 'output_labels' folder
	* 'test_labels.csv' and 'train_labels.csv' can be converted to tfrecord files for TF


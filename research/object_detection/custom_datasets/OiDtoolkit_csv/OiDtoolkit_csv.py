import os
import numpy as np
import pandas as pd
import cv2


path_input_images = 'input_images/'
path_input_txts = 'input_txts/'
path_output_labels = 'output_labels/'


# configuration
# maximum resolution
max_pixels = 600
# train/test ratio
all_images_list = sorted([name for name in os.listdir(path_input_images)])
train_batch = 900
test_batch = 100
# creates list of images using the first 'train_batch', and the last 'test_batch' images
images_list = np.append(all_images_list[0:train_batch], all_images_list[-test_batch:])
print(f"Analizing {len(all_images_list)} images, using first {train_batch} for training and last {test_batch} for test")



with open(f"{path_output_labels}train_labels.csv", 'w') as output_labels_file_train, open(f"{path_output_labels}test_labels.csv", 'w') as output_labels_file_test:
	output_labels_file_train.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
	output_labels_file_test.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")


	for i, image_name in enumerate(images_list):

		if i%50==0 : print(f"\tChecking image nr. {i}")


		if i<train_batch:
			subset = 'train'
			out_labels_file = output_labels_file_train
		else:
			subset = 'test'
			out_labels_file = output_labels_file_test


		img = cv2.imread(path_input_images + image_name, 1)
		height, width, channel = img.shape


		# resize images if their biggest size is > 600px
		factor = max(width, height)/max_pixels if (max(width, height)) > max_pixels else 1
		width, height = int(width/factor), int(height/factor)
		img = cv2.resize(img, (width, height))


		cv2.imwrite(f"output_images_{subset}/{image_name}", img)

		with open(f"{path_input_txts}{os.path.splitext(image_name)[0]}.txt", 'r') as txt_file:
			df = pd.read_csv(txt_file, sep='\s+', header=None, index_col=None)
			# print(df)

			for i in range(df.shape[0]):
				label = df.iloc[i,0].lower()
				x_min = int(df.iloc[i,1]/factor)
				y_min = int(df.iloc[i,2]/factor)
				x_max = int(df.iloc[i,3]/factor)
				y_max = int(df.iloc[i,4]/factor)



				out_labels_file.write(f"{image_name},{width},{height},{label},{x_min},{y_min},{x_max},{y_max}\n")

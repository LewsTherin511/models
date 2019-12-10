import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET



def xml_to_csv(path):
	xml_list = []
	# go through xml files in the directory
	for xml_file in glob.glob(path + '/*.xml'):
		tree = ET.parse(xml_file)
		root = tree.getroot()
		for member in root.findall('object'):
			value = (root.find('filename').text,
					 int(root.find('size')[0].text),
					 int(root.find('size')[1].text),
					 member[0].text,
					 int(member[4][0].text),
					 int(member[4][1].text),
					 int(member[4][2].text),
					 int(member[4][3].text)
					 )
			xml_list.append(value)
	column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
	xml_df = pd.DataFrame(xml_list, columns=column_name)
	return xml_df


def main():
	input_data_path = "input_images_xml/"
	output_csv_path = "output_csv/"
	for subset in ['train', 'test']:
		xml_files_path = os.path.join(os.getcwd(), f'{input_data_path}{subset}')
		xml_df = xml_to_csv(xml_files_path)
		xml_df.to_csv(f'{output_csv_path}{subset}_labels.csv', index=None)
		print(f'Successfully converted xml to csv for {subset} images.')


main()

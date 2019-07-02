import pickle
import code
import os,cv2
import numpy as np
def open_file_to_check_dim():
	for i in range(9):
		folder_num = "0"+str(i+1)
		print(folder_num)
		folder_name = "F:/Dropbox/git prjects/MRNet/vol" + folder_num
		sub_file_list = os.listdir(folder_name)

		for item_name in sub_file_list:
			data = pickle.load(open(folder_name + "/" + item_name,"rb"))
			print(data.shape)
def make_pck_file_from_pngs():
	folder_name="F:/Dropbox/0.2018-1/2.SMC/shoulder/MRI_T2_COR_2019_02_20"
	out_folder= "F:/Dropbox/git prjects/MRNet/sholder20190702"
	size = 320
	for sub_folder_name in os.listdir(folder_name):
		slides = []
		item_name_list = os.listdir(folder_name+"/"+ sub_folder_name)
		for item_name in item_name_list:
			img = cv2.imread(folder_name+"/"+ sub_folder_name+"/"+item_name,0)
			img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
			img = img.astype(np.int16)
			slides.append(img)

		out_path = out_folder+"/"+sub_folder_name+".pck"
		slides = np.asarray(slides)
		pickle.dump(slides, open(out_path,"wb"))

def pck_file_train_test_split():
	from sklearn.model_selection import train_test_split

	in_folder= "F:\\Dropbox\\git prjects\\MRNet\\sholder20190702"
	data = os.listdir(in_folder)
	labels = os.listdir(in_folder)
	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.4, random_state=42)
	data_validation, data_test, labels_validation, labels_test  = train_test_split(data_test, labels_test, test_size=0.5, random_state=42)

	mapping = {"train":labels_train,"validation":labels_validation,"test":labels_test}

	out_path = "F:\\Dropbox\\git prjects\\MRNet\\shoulder_dataset"
	in_path = in_folder

	for data_type, file_name_list in mapping.items():
		print(data_type)
		for file_name in file_name_list:
			cmd = 'copy "' + in_path+'\\'+file_name + '" "' + out_path + '\\'+data_type+'"'		 
			os.system(cmd)

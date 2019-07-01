import pickle
import code
import os


for i in range(9):
	folder_num = "0"+str(i+1)
	print(folder_num)
	folder_name = "F:/Dropbox/git prjects/MRNet/vol" + folder_num
	sub_file_list = os.listdir(folder_name)

	for item_name in sub_file_list:
		data = pickle.load(open(folder_name + "/" + item_name,"rb"))
		print(data.shape)
code.interact(local=dict(globals(), **locals()))
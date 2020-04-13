# !/usr/bin/python
# _*_ coding:utf8 _*_
'''
description: load trained model to recognize Digital number
code date: 2018/08/02
modify date: 2018/08/21
athor: TTang
'''

import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
import os
import re

def start():
	print("################Testing starting#####################")
	# load testing data
	print("################Loading testing data#####################")
	x_test=np.load(os.getcwd()+'/datasets/X_test.npy')
	y_test=np.load(os.getcwd()+'/datasets/Y_test.npy')

	# x_test = x_test.reshape(x_test.shape[0],-1)/255.0
	# y_test = np_utils.to_categorical(y_test,num_classes=10)

	# load trained model
	print("################Loading trained model#####################")
	model = load_model(os.getcwd()+'/models/model_HSI_recognition.h5')

	# test model
	print("################Testing model#####################")
	loss,accuracy = model.evaluate(x_test,y_test)

	print("################Testing Result#####################")
	print('test loss',loss)
	print('accuracy',accuracy)
	print("################Testing End#####################")


	# write structure of model into file
	print("################Printing structure of model in json form#####################")
	from keras.models import model_from_json
	json_string = model.to_json()
	model = model_from_json(json_string)

	txtPath=os.getcwd()+'/descriptionOfmodel.txt'
	txtFile=open(txtPath,'w')
	txtFile.write(' Description of trained model ')
	txtFile.write('\n')
	description=re.split('[{}\[\]]',json_string)
	for content in description:
		txtFile.write(content)
		txtFile.write('\n')
	txtFile.write(' Description of trained model ')
	txtFile.close()

	print(' The description of model has been written in file：\n %s'%txtPath)

if __name__=='__main__':
	start()
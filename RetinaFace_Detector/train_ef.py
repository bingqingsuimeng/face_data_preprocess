import time
import numpy as np
import argparse
import cv2
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import face_preprocess
from efficientnet_pytorch import EfficientNet



class Race_Detect(object):
	"""docstring for ClassName"""
	def __init__(self, ckpt_path):
		super(ClassName, self).__init__()
		self.MODEL=EfficientNet.from_pretrained('efficientnet-b1',num_classes=4)

		if torch.cuda.device_count() > 1:
			self.MODEL = nn.DataParallel(MODEL)
		if torch.cuda.is_available():
			self.MODEL.cuda()
			print('GPU count: %d'%torch.cuda.device_count())
			print('CUDA is ready')
		
		self.MODEL.load_state_dict(torch.load(self.ckpt_path+'model_8_135.pth.tar'))

		self.MODEL.eval()

		self.classes=['White','African','Asian','Indian']


	def detect(img,point):
		face_preprocess.preprocess(img,point)
		img = cv2.resize(128,128)
		img = Variable(img).cuda()
		output = self.MODEL(img)
		predit=torch.argmax(output,dim=1)

		return self.classes[predit]

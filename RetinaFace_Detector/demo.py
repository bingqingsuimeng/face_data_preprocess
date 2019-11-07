import sys
sys.path.append('./util/')
import face_model
import cv2
import numpy as np
import scipy.io as sio 
import os
import glob
import cv2
from RetinaFace_Detector.retinaface import RetinaFace
import time
import race

os.environ["CUDA_VISIBLE_DEVICES"]='0'

thresh = 0.8
scales = [1024, 1980]
im_shape = [1024,1980,3]
count = 1
gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
#im_scale = 1.0
#if im_size_min>target_size or im_size_max>max_size:
im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)
print('im_scale', im_scale)
scales = [im_scale]
flip = False

race=race.Race_Detect(ckpt_path='./efb1/')

def drawing_landmark(image,landmark):
        #if landmark == None: return None
        radius = 3
        color  = (0,255,0)
        thickness = -1
        for x,y in landmark:
            cv2.circle(image,(x,y),radius,color, thickness)

def run():
	vidcap = cv2.VideoCapture(0)
	vidcap.set(3,1024)
	vidcap.set(4,768)

	#vidcap = cv2.VideoCapture('2019-07-10-170304.webm')
	success,image = vidcap.read()
	success = True
	while success:
		# try:
			global frame_number
			success,image = vidcap.read()
			t0=time.time()
			#cv2.imwrite("Images/%d.jpg" % start, img)     # save frame as JPEG file
			#bbox,points=model.get_faces(image)
			bbox, points = detector.detect(image, thresh, scales=scales, do_flip=flip)
			# print("time",time.time()-t0)
			if bbox is not None and bbox.shape[0]!=0:
				for i in range(bbox.shape[0]):
					point= points[i]
					bbox1=bbox[i,0:4]
					score= bbox[i,4]
					#drawing_landmark(image,point)
					# print(image.shape)
					print(race.detect(img=image, point=point))
					cv2.rectangle(image,(int(bbox1[0]),int(bbox1[1])),(int(bbox1[2]),int(bbox1[3])),(0,255,128),2)

			cv2.imshow('Object detector', image)
			# Press 'q' to quit
			if cv2.waitKey(1) == ord('q'):
				break


	print('finish')
	# Clean up
	vidcap.release()
	cv2.destroyAllWindows()


run()
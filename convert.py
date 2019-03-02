import cv2
import numpy as np
import os


dir= os.listdir('./sf_data')
count=0
global_count=0
if not os.path.exists('./train/'):
	os.mkdir('./train')
for folder in dir:
	print('Making class')
	classes= os.listdir('./sf_data/'+folder)
	#count=0
	if not os.path.exists('./train/'+folder):
		os.mkdir('./train/'+folder)
	#matrix=np.zeros((10,240,180))

	for videos in classes:
		matrix=np.zeros((5,240,180))
		count=0
		for img in videos:
			frame=cv2.imread(img,0)	
			matrix[count]=frame
			count+=1
			if(count%5==0):
				break

		np.save('./train/'+folder+'/'+str(global_count),matrix)
		global_count+=1
		cv2.destroyAllWindows()
	#count+=1
	#if(count==5):
	#	break
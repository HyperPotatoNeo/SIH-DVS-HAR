import cv2
import numpy as np
import os

FOLDERNAME=?

dir= os.listdir('./'+FOLDERNAME)

global_count=0

for folder in dir:
	os.mkdir('./train/'+folder)
	inner_dir= os.listdir('./'+folder)
	count=0
	matrix=np.zeros((10,?,?))

	for videos in inner_dir:
		cap = cv2.VideoCapture(videos)

		while(cap.isOpened()):
			ret, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			matrix[count]=frame
			count+=1
			if(count%10==0)
			break

		np.save('./train'+folder+str(global_count),matrix)
		global_count+=1
		cap.release()
		cv2.destroyAllWindows()
import os
import pickle
from PIL import Image

import cv2

if __name__ == "__main__":
	pth = "/media/forsad/grabell_box2/study_crops/1001/1001_COLOR_0 Video 1 4_4_2018 2_42_38 PM 1.pickle"
	with open(pth, "rb") as f:
		dt = pickle.load(f)
		for crops in dt:
			for crop in crops:
				#print(crop.shape)
				#print(crop.dtype)
				#im = Image.fromarray(crop)
				#im.show()
				cv2.imshow("crop", crop)
				cv2.waitKey(0)
				#exit(0)
import cv2

vcap = cv2.VideoCapture('C:/Multispectral Data/7-23-19-rgb/DSC_0001.MOV') # 0=camera

if vcap.isOpened(): 
    # get vcap property 
	width = vcap.get(3)
	height = vcap.get(4)
	fps = vcap.get(5)
	total = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(width)
	print(height)
	print(fps)
	print(total)
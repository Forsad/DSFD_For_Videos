from skimage import io
import numpy as np
import os
import cv2
import csv

import torch
from face_ssd_infer import SSD
from utils import vis_detections


device = torch.device("cpu")
conf_thresh = 0.3
target_size = (800, 800)

net = SSD("test")
net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth', map_location='cpu'))
net.to(device).eval();

#Set folder paths
folder = "C:/Multispectral Data/7-23-19-rgb"          #set folder path

file_type = 5 # 0- tiff videos,5- rgb videos,1- tiff image sequence 
wavelength_map = {
    (0, 4): '975',
    (0, 3): '960',
    (0, 2): '945',
    (0, 1): '930',
    (0, 0): '915',
    (1, 4): '900',
    (1, 3): '890',
    (1, 2): '875',
    (1, 1): '850',
    (1, 0): '835',
    (2, 4): '820',
    (2, 3): '805',
    (2, 2): '790',
    (2, 1): '775',
    (2, 0): '760',
    (3, 4): '745',
    (3, 3): '730',
    (3, 2): '715',
    (3, 1): '700',
    (3, 0): '675',
    (4, 4): '660',
    (4, 3): '645',
    (4, 2): '630',
    (4, 1): '615',
    (4, 0): '600'
}

#Multispectral video data processing settings
frame_discard_start = 0
frame_discard_end = 0

#Get list of all files in the given directory.
def getFilesList(dirName):
    listOfFile = sorted(os.listdir(dirName))
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        allFiles.append(fullPath)
        #print("Full path ",fullPath)
    return allFiles

def getImageSequence(dirName='Testing'):
    listOfFile = [dI for dI in sorted(os.listdir(dirName)) if os.path.isdir(os.path.join(dirName,dI))]
    #listOfFile = sorted(os.listdir(dirName))
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        allFiles.append(fullPath)
    print("All the files:",allFiles)
    return allFiles

#Intermediate function to call the get all files paths from the directory
def getVideoFileList():
    global  folder
    videos_paths = getFilesList(folder)
    return  videos_paths

# When only one multispectral data needs to be analysed
def singleVideoFile():
    raw = io.imread(folder + "")

#readVideoFile() reads the video file and returns the mean of every available wavelength for each frame present and also the normalizes the frequencies across frame
#The frequencies are mapped in ascending order to the array index
def  readVideoFile(video_path,name):
    global frame_discard_end
    global frame_discard_start
    global wavelength_noise_mean

    raw = video_path#io.imread(video_path)
    print("TOTAL NO OF FRAMES:", raw.shape[0])
    raw_io = raw[frame_discard_start:(raw.shape[0]-frame_discard_end),:]
    frames = raw_io.shape[0]

    wavelength_frame_avg = np.zeros(shape=(25,frames))
    wavelength_normalized = np.zeros(shape=(25,frames))
    wavelength_sums = np.zeros(shape=(25))

    for i in range(0,frames):
        data = raw_io[i,]
        for j in wavelength_map.keys():
            index = 24-(j[0]*5)+(j[1]-4)
            wavelength_frame_avg[index,i] = np.mean(data[j[0]::5,j[1]::5])  #map frequencies in increasing order of the array ie., index 0 will have 600...index 24 will have 975
    wavelength_frame_avg = wavelength_frame_avg.transpose()
    writeCSVFile(wavelength_frame_avg,name)

def  readSequenceFile(wavelength_frame_avg,name):
    wavelength_frame_avg = wavelength_frame_avg.transpose()
    writeCSVFile(wavelength_frame_avg,name)
    
def readSequenceData(folder):
    images = []
    wavelength_frame_avg = np.zeros(shape=(25,len(os.listdir(folder))))
    i=0
    for filename in sorted(os.listdir(folder)):
        data = cv2.imread(os.path.join(folder,filename),cv2.COLOR_BGR2GRAY)
        #print("filename:",filename," data:",data.shape," ",data.shape)
        for j in wavelength_map.keys():
            index = 24-(j[0]*5)+(j[1]-4)
            wavelength_frame_avg[index,i] = np.mean(data[j[0]::5,j[1]::5])  #map frequencies in increasing order of the array ie., index 0 will have 600...index 24 will have 975
        print("i:",i)
        i = i+1
    print(" images:",filename)
    return wavelength_frame_avg

def multipleVideoFiles():
    if file_type==0:
    	video_files = getVideoFileList()
    elif file_type==5:
        video_files = getVideoFileList()
    else:
        video_files = getImageSequence(folder)
    for video_id in range(0,len(video_files)):
        #print(psutil.virtual_memory())
        print("video id:",video_files[video_id])
        if file_type==0:
             raw = io.imread(video_files[video_id])
             readVideoFile(raw,video_files[video_id])
        elif file_type==5:
            r =[]
            g = []
            b =[]
            cnt=0
            #print("Rgb list:",video_files[video_id])
            cap = cv2.VideoCapture(video_files[video_id])
            while(cap.isOpened()):
                ret, frame = cap.read()
                detections = net.detect_on_image(frame, target_size, device, is_pad=False, keep_thresh=conf_thresh)
                if detections.size > 0:
                    bbox = [int(i) for i in detections[0][0:4]]
                    height = bbox[3] - bbox[1]
                    width = bbox[2] - bbox[0]
                    cropped_image = frame[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
                else:
                    cropped_image = frame
                # cropped_image = img_data[(bbox[1] + int(0.1*height)): (bbox[1] + int(0.2*height)), (bbox[0] + int(0.3*width) ) : (bbox[0] + int(0.7*width))]
                
                print("_cnt:",cnt)
                cnt=cnt+1
                if ret == True:
                    #print("RGB shape:",frame.shape)
                    b.append(np.mean(cropped_image[:,:,0]))
                    g.append(np.mean(cropped_image[:,:,1]))
                    r.append(np.mean(cropped_image[:,:,2]))
                # Break the loop
                else: 
                    break
             
            cap.release()
            data = np.column_stack((r,g,b))
            #writeCsvFile1(r,g,b,video_files[video_id])
            writeCSVFile(data,video_files[video_id])
             
             
             #print("RGB shape:",raw)
        else:
             print("video files:",video_files[video_id])
             raw = readSequenceData(video_files[video_id])
             readSequenceFile(raw,video_files[video_id])
        raw=0
def writeCsvFile1(r,g,b,name):
    names = name.split('\\')[1].split('.')
    myFile = open("storecsv/22-7-2019-rgb/"+names[0]+'.csv', 'w', newline='')
    with myFile:
        wr = csv.writer(myFile)
        wr.writerow(r)
        wr.writerow(g)
        wr.writerow(b)
        
# Write CSV data to the file
def writeCSVFile(data,name):
    print("Name,",name)
    if file_type==0:
        splits = name.split("/")
        names = splits[len(splits)-1].split(".")
        myFile = open("storecsv/rgb/"+names[0]+'.csv', 'w', newline='')
    elif file_type == 5:
        names = name.split('\\')[1].split('.')
        myFile = open("storecsv/rgb/"+names[0]+'.csv', 'w', newline='')    
    else:
        names = name.split('\\')
        myFile = open("storecsv/rgb/"+names[1]+'.csv', 'w', newline='')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)
	
###########################MAIN###################################
multipleVideoFiles()


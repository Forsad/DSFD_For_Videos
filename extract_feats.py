import cv2
import torch

#from face_ssd_infer import SSD
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
import torch.nn as nn
from data.config import TestBaseTransform, widerface_640 as cfg
from layers import Detect, get_prior_boxes, FEM, pa_multibox, mio_module, upsample_product
from utils import resize_image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import pickle
from face_ssd_infer import SSD

def eucledian(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def centroid(bbox):
    return (bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0)

def vis_detections_cur(im, dets, fig, ax, history, thresh=0.5, show_text=True):
    """Draw detected bounding boxes."""
    class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0] if dets is not None else []
    if len(inds) == 0:
        return []
    im = im[:, :, (2, 1, 0)]
    #plt.clf()
    [p.remove() for p in reversed(ax.patches)]
    ax.imshow(im, aspect='equal')
    #print(dets)
    #exit(0)
    cur_history = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cur_centroid = centroid(bbox)
        #print(cur_centroid)
        #for cent in history:
        #if eucledian(cent, cur_centroid) < 1000 or True:
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2.5))
        #break
    return cur_history

def process_img(img, target_size, device, conf_thresh, net):
    (all_detections, keep_idxes) = net.detect_on_image(img, target_size, device, False, conf_thresh)
    #print(detections)
    #print(type(detections))
    keep_idxes = all_detections[:, 4] > conf_thresh
    #print(all_detections)
    #print(keep_idxes)
    detections = all_detections[keep_idxes, :]
    #exit(0)
    #print(detections[0])
    for idx in range(detections.shape[0]):
        bbox = detections[idx, :4]
        ibbox = [int(round(x)) for x in bbox]
        #exit(0)
        cv2.rectangle(img, (ibbox[0], ibbox[1]), (ibbox[2], ibbox[3]), (0, 255, 0), 5)
        #exit(0)

        #crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    #print(bbox)
    # cv2.imshow("cropped", img)
    # cv2.waitKey(0)
    # exit(0)
    #cv2.imwrite("out2.png", crop_img)
    return all_detections
    #cv2.waitKey(0)

    #history = vis_detections_cur(img, detections, fig, ax, history, conf_thresh, show_text=False)
    #return history

def video_cap_and_process(vidcap, target_size, device, conf_thresh, net, cvWriter):
    
    success,image = vidcap.read()
    # plt.ion()
    # fig, ax = plt.subplots(figsize= (12, 12))
    cnt = 0
    # history = []
    embeds = []
    while success:
        embed = process_img(image, target_size, device, conf_thresh, net)
        cvWriter.write(image)
        #cv2.imwrite("out3.png", embed[0])
        embeds.append(embed)
        cnt += 1
        if cnt%(60 * 30) == 0:
            break
        #if cnt%(60*60) == 0:
            #print("Done with " + str(cnt/(60 * 60)) + " minutes")
        #print(cnt)
        #cv2.imshow(image)     # save frame as JPEG file

        success, image = vidcap.read()
    return embeds
        #process_img(image, target_size, device, conf_thresh)

def video_cap_for_file(fl, device, out_fl, net):
    try:
        conf_thresh = 0.3
        out_fll = out_fl[:-4] + ".pickle"
        if os.path.exists(out_fll):
            print("File exists: " + out_fll)
            return
        cap = cv2.VideoCapture(fl)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        target_size = (w, h)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cvWriter = cv2.VideoWriter(out_fl, fourcc, fps, (int(w), int(h)))

        ret = video_cap_and_process(cap, target_size, device, conf_thresh, net, cvWriter)
        with open(out_fl[:-4] + ".pickle","wb") as ff:
            pickle.dump(ret, ff)
            ff.close()
        cap.release()
        cvWriter.release()
        print('cvWriter done')
        print("Done with " + fl)
    except Exception as e:
        print(str(e) + "; was working on " + fl)
    #print(len(ret))
    #print(len(ret[450]))
    #cv2.imwrite("out5.png", ret[450][0])
    #exit(0)

if __name__ == "__main__":
    #fl = "/media/forsad/Expansion_3/Study Components/FACS/ICK Videos for FACS/1001/1001_COLOR_0 Video 2 4_4_2018 2_42_38 PM 2.mp4"
    num_devices = 1
    nets = []
    for gpu_no in range(num_devices):
        device = torch.device("cuda:" + str(gpu_no))
        net = SSD("test:" + str(gpu_no))
        net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
        net.to(device).eval()
        nets.append(net)
    #base_dir = "/media/forsad/grabell_box2/Study Components/FACS/ICK Videos for FACS"
    base_dir = "/media/forsad/Seagate Expansion Drive/Study Components/FACS/ICK Videos for FACS"
    folders = os.listdir(base_dir)

    out_dir = "/media/forsad/Seagate Expansion Drive/study_crops_test"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_fls = []
    for folder in folders:
        #print(folder)
        full_folder = os.path.join(base_dir, folder)
        if not os.path.isdir(full_folder):
            continue
        #print(foler)
        out_folder = os.path.join(out_dir, folder)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        fls = os.listdir(full_folder)
        for fl in fls:
            if not fl.endswith(".mp4"):
                continue
            full_p = os.path.join(full_folder, fl)
            out_fl = os.path.join(out_folder, fl)
            all_fls.append((full_p, out_fl))
    with ThreadPoolExecutor(max_workers=num_devices):
        for idx, fl in enumerate(all_fls):
            full_p = fl[0]
            out_fl = fl[1]
            gpu_no = idx%num_devices
            video_cap_for_file(full_p, device, out_fl, nets[gpu_no])
                #fl = "/media/forsad/grabell_box2/Study Components/FACS/ICK Videos for FACS/1001/1001_FETCH_0 Video 1 4_4_2018 2_16_04 PM 1.mp4"
    #video_cap_for_file(fl, device, out_dir)
    # conf_thresh = 0.3
    # cap = cv2.VideoCapture(fl)
    # w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # target_size = (w, h)


  
    #self.resnet = models.resnet18(pretrained=True)
    #embeds = video_cap_and_process(cap, target_size, device, conf_thresh)
    #print(type(img))
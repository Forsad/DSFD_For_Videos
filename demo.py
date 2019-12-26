import cv2
import torch

from face_ssd_infer import SSD
import numpy as np
import matplotlib.pyplot as plt
import math

def eucledian(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def centroid(bbox):
    return (bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0)

def vis_detections_cur(im, cnt, dets, fig, ax, history, thresh=0.5, show_text=True):
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
        cur_history.append(centroid(bbox))
        if show_text:
            ax.text(bbox[0], bbox[1] - 5,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=10, color='white')
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("samp/" + str(cnt) + ".jpg")
    plt.show(block=False)
    plt.pause(.1)
    return cur_history

def process_img(img, cnt, target_size, device, conf_thresh, fig, ax, history):
    detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
    history = vis_detections_cur(img, cnt, detections, fig, ax, history, conf_thresh, show_text=False)
    return history

def video_cap(vidcap, target_size, device, conf_thresh):
    
    success,image = vidcap.read()
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 12))
    cnt = 0
    history = []
    while success:
        if cnt % 60 == 0:
            history = process_img(image, cnt, target_size, device, conf_thresh, fig, ax, history)
        #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file

        success,image = vidcap.read()
        cnt += 1
        #process_img(image, target_size, device, conf_thresh)
def video_cap_and_process(vidcap, target_size, device, conf_thresh):
    
    success,image = vidcap.read()
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(12, 12))
    # cnt = 0
    # history = []
    while success:
        if cnt % 10 == 0:
            history = process_img(image, target_size, device, conf_thresh, fig, ax, history)
        #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file

        success,image = vidcap.read()
        cnt += 1
        #process_img(image, target_size, device, conf_thresh)



    
if __name__ == "__main__":
    #fl = "/media/forsad/Expansion_3/Study Components/FACS/ICK Videos for FACS/1001/1001_COLOR_0 Video 2 4_4_2018 2_42_38 PM 2.mp4"
    device = torch.device("cuda")
    net = SSD("test")
    net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
    net.to(device).eval()

    #fl = "/media/forsad/Expansion_3/Study Components/FACS/ICK Videos for FACS/1012/1012_COLOR_0 Video 1 5_16_2018 10_15_30 AM 1.mp4"
    fl = "/media/forsad/grabell_box2/Study Components/FACS/ICK Videos for FACS/1029/1029_ICK_B_0 Video 2 7_30_2018 3_50_01 PM 2.mp4"
    cap = cv2.VideoCapture(fl)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    target_size = (w, h)
    video_cap(cap, target_size, device, 0.3)

    

    #print(type(img))
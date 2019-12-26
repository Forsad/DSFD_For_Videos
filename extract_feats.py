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
import os
import pickle


class SSD(nn.Module):

    def __init__(self, phase, nms_thresh=0.3, nms_conf_thresh=0.01):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = 2
        self.cfg = cfg

        resnet = torchvision.models.resnet152(pretrained=True)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(
            *[nn.Conv2d(2048, 512, kernel_size=1),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True)]
        )
        self.layer6 = nn.Sequential(
            *[nn.Conv2d(512, 128, kernel_size=1, ),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True)]
        )

        output_channels = [256, 512, 1024, 2048, 512, 256]

        # FPN
        fpn_in = output_channels

        self.latlayer3 = nn.Conv2d(fpn_in[3], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(fpn_in[2], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(fpn_in[1], fpn_in[0], kernel_size=1, stride=1, padding=0)

        self.smooth3 = nn.Conv2d(fpn_in[2], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(fpn_in[1], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(fpn_in[0], fpn_in[0], kernel_size=1, stride=1, padding=0)

        # FEM
        cpm_in = output_channels

        self.cpm3_3 = FEM(cpm_in[0])
        self.cpm4_3 = FEM(cpm_in[1])
        self.cpm5_3 = FEM(cpm_in[2])
        self.cpm7 = FEM(cpm_in[3])
        self.cpm6_2 = FEM(cpm_in[4])
        self.cpm7_2 = FEM(cpm_in[5])

        # head
        head = pa_multibox(output_channels)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        if self.phase != 'onnx_export':
            self.detect = Detect(self.num_classes, 0, cfg['num_thresh'], nms_conf_thresh, nms_thresh,
                                 cfg['variance'])
            self.last_image_size = None
            self.last_feature_maps = None

        if self.phase == 'test':
            self.test_transform = TestBaseTransform((104, 117, 123))

    def forward(self, x):

        image_size = [x.shape[2], x.shape[3]]
        loc = list()
        conf = list()

        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)

        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)
        #print(conv7_2_x.size())
        lfpn3 = upsample_product(self.latlayer3(fc7_x), self.smooth3(conv5_3_x))
        lfpn2 = upsample_product(self.latlayer2(lfpn3), self.smooth2(conv4_3_x))
        lfpn1 = upsample_product(self.latlayer1(lfpn2), self.smooth1(conv3_3_x))

        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1

        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]

        sources[0] = self.cpm3_3(sources[0])
        sources[1] = self.cpm4_3(sources[1])
        sources[2] = self.cpm5_3(sources[2])
        sources[3] = self.cpm7(sources[3])
        sources[4] = self.cpm6_2(sources[4])
        sources[5] = self.cpm7_2(sources[5])

        # apply multibox head to source layers
        featuremap_size = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([x.shape[2], x.shape[3]])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            len_conf = len(conf)
            cls = mio_module(c(x), len_conf)
            conf.append(cls.permute(0, 2, 3, 1).contiguous())

        face_loc = torch.cat([o[:, :, :, :4].contiguous().view(o.size(0), -1) for o in loc], 1)
        face_loc = face_loc.view(face_loc.size(0), -1, 4)
        face_conf = torch.cat([o[:, :, :, :2].contiguous().view(o.size(0), -1) for o in conf], 1)
        face_conf = self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes))

        if self.phase != 'onnx_export':

            if self.last_image_size is None or self.last_image_size != image_size or self.last_feature_maps != featuremap_size:
                self.priors = get_prior_boxes(self.cfg, featuremap_size, image_size).to(face_loc.device)
                self.last_image_size = image_size
                self.last_feature_maps = featuremap_size
            with torch.no_grad():
                output = self.detect(face_loc, face_conf, self.priors)
        else:
            output = torch.cat((face_loc, face_conf), 2)
        return output

    def detect_on_image(self, source_image, target_size, device, is_pad=False, keep_thresh=0.3):

        image, shift_h_scaled, shift_w_scaled, scale = resize_image(source_image, target_size, is_pad=is_pad)

        x = torch.from_numpy(self.test_transform(image)).permute(2, 0, 1).to(device)
        x.unsqueeze_(0)

        detections = self.forward(x).cpu().numpy()
        #print(detections)

        scores = detections[0, 1, :, 0]
        keep_idxs = scores > keep_thresh  # find keeping indexes
        detections = detections[0, 1, keep_idxs, :]  # select detections over threshold
        detections = detections[:, [1, 2, 3, 4, 0]]  # reorder

        detections[:, [0, 2]] -= shift_w_scaled  # 0 or pad percent from left corner
        detections[:, [1, 3]] -= shift_h_scaled  # 0 or pad percent from top
        detections[:, :4] *= scale

        return detections


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

def process_img(img, target_size, device, conf_thresh):
    detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
    #print(detections)
    imgs = []
    #print(type(detections))
    for idx in range(detections.shape[0]):
        bbox = detections[idx, :4]
        bbox = [int(round(x)) for x in bbox]
        #print(bbox)
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        imgs.append(crop_img)
    #print(bbox)
    #cv2.imshow("cropped", crop_img)
    #cv2.imwrite("out2.png", crop_img)
    return imgs
    #cv2.waitKey(0)

    #history = vis_detections_cur(img, detections, fig, ax, history, conf_thresh, show_text=False)
    #return history

def video_cap_and_process(vidcap, target_size, device, conf_thresh):
    
    success,image = vidcap.read()
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(12, 12))
    cnt = 0
    # history = []
    embeds = []
    while success:
        if cnt % 10 == 0:
            embed = process_img(image, target_size, device, conf_thresh)
            #cv2.imwrite("out3.png", embed[0])
            embeds.append(embed)
        cnt += 1
        if cnt%(60*60) == 0:
            print("Done with " + str(cnt/(60 * 60)) + " minutes")
        #print(cnt)
        #cv2.imshow(image)     # save frame as JPEG file

        success,image = vidcap.read()
    return embeds
        #process_img(image, target_size, device, conf_thresh)

def video_cap_for_file(fl, device, out_fl):
    conf_thresh = 0.3
    out_fll = out_fl[:-4] + ".pickle"
    if os.path.exists(out_fll):
        print("File exists: " + out_fll)
        return
    cap = cv2.VideoCapture(fl)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    target_size = (w, h)
    ret = video_cap_and_process(cap, target_size, device, conf_thresh)
    with open(out_fl[:-4] + ".pickle","wb") as ff:
        pickle.dump(ret, ff)
        ff.close()
    #print(len(ret))
    #print(len(ret[450]))
    #cv2.imwrite("out5.png", ret[450][0])
    #exit(0)

if __name__ == "__main__":
    #fl = "/media/forsad/Expansion_3/Study Components/FACS/ICK Videos for FACS/1001/1001_COLOR_0 Video 2 4_4_2018 2_42_38 PM 2.mp4"
    
    device = torch.device("cuda")
    net = SSD("test")
    net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
    net.to(device).eval()

    base_dir = "/media/forsad/grabell_box2/Study Components/FACS/ICK Videos for FACS"
    folders = os.listdir(base_dir)

    out_dir = "/media/forsad/grabell_box2/study_crops"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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

            print(full_p)
            out_fl = os.path.join(out_folder, fl)
            video_cap_for_file(full_p, device, out_fl)
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
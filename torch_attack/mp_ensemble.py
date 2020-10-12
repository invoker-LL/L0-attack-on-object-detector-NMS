# coding=utf-8
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import sys
sys.path.append('../mmdetection/')
sys.path.append('../../adversial_od/') #注意需要添加绝对路径
from mmdet.apis import init_detector, inference_detector_with_grad, data_loader_mmdet
from tqdm import tqdm
import kmeans_pixel_metric as kmeans
from PIL import Image
from utils.utils import *
import cv2
def pixel_coordinates_checker(i,j):
    if i<0 or j < 0 or i >=500 or j >= 500:
        return False
    return True

def set_val(i, j, temp_mask):
    if pixel_coordinates_checker(i,j):
        temp_mask[0, i, j] = True
    else:
        #print('pixel index error')
        pass
    return temp_mask

def connect_points(temp_mask,edges_per_img):
# this function need to be faster
    for edge in edges_per_img:
        cnt_plot = 0
        ori_point,ori_center = np.round(edge[0]).astype(np.int),np.round(edge[1]).astype(np.int)
        #temp_mask[0, ori_point[0], ori_point[1]] = True
        distance = ori_point-ori_center
        delta_x = distance[0]
        delta_y = distance[1]
        abs_x = abs(delta_x)
        abs_y = abs(delta_y)
        up_right,up_left,bt_right,bt_right = False,False,False,False
        #x越小越上
        #y越小越左
        if delta_x < 0:  # 上
            if delta_y >= 0:  # 右
                up_right = True
            else:  # 左
                up_left = True
            center = ori_center
            point = ori_point
        else:  # 下
            if delta_y >= 0:  # 右
                up_left = True
            else:
                up_right = True
            center = ori_point
            point = ori_center
        step2 = abs(abs_x-abs_y)
        if up_right:
            if abs_x>abs_y:
                j = center[1]
                for i in range(center[0],center[0]-1-step2,-1):

                    temp_mask = set_val(i,j,temp_mask)#竖走直线

                i = center[0]-1-step2
                for j in range(center[1],point[1]+1):

                    temp_mask = set_val(i,j,temp_mask)#竖走直线
                    i-=1
            else:
                i = center[0]
                for j in range(center[1], center[1] + 1 + step2):  # j越大越右

                    temp_mask = set_val(i,j,temp_mask)#竖走直线
                j = center[1] + 1 + step2
                for i in range(center[0] - 1, point[0] - 1, -1):  # i越小越上

                    temp_mask = set_val(i,j,temp_mask)#竖走直线
                    j += 1
        if up_left:
            if abs_x>abs_y:
                j = center[1]
                for i in range(center[0],center[0]-1-step2,-1): #横走直线，左

                    temp_mask = set_val(i,j,temp_mask)#竖走直线
                i = center[0]-1-step2
                for j in range(center[1],point[1]-1,-1): #j越小越左

                    temp_mask = set_val(i,j,temp_mask)#竖走直线
                    i-=1
            else:
                i = center[0]
                for j in range(center[1],center[1]-1-step2,-1):

                    temp_mask = set_val(i,j,temp_mask)#竖走直线
                j = center[1]-1-step2
                for i in range(center[0],point[0]-1,-1):

                    temp_mask = set_val(i,j,temp_mask)#竖走直线
                    j-=1
    return temp_mask

def pixel_dist_metric(p1,p2):
    delta_x = abs(p1[0]-p2[0])
    delta_y = abs(p1[1]-p2[1])
    return np.maximum(delta_x,delta_y)

class Gen_edges_with_kmeans_recursively():
    def __init__(self):
        self.edges_per_img = []
    def get_final_edges(self):
        return self.edges_per_img

    def connect_nearest_to_mid(self,clusters,centers):
        mid_center = (centers[0] + centers[1] + centers[2])/3
        mid_center = np.round(mid_center)
        nearest_points = np.zeros_like(centers)
        j = 0
        for clster in clusters:
            min_dist = 608
            for idx in clster:
                temp_dist = pixel_dist_metric(idx,mid_center)
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    nearest_points[j] = idx
            j+=1
        for j in range(3):
            self.edges_per_img.append([mid_center, nearest_points[j]])
        return

    def recursive_get_edges(self,clusters1, centers1):

        for i in range(len(centers1)):
            center1_i = centers1[i]
            cluster_i = clusters1[i]
            len_clster = len(cluster_i)
            if len_clster == 1:
                self.edges_per_img.append([cluster_i[0], cluster_i[0]])
            elif len_clster == 2:
                self.edges_per_img.append([cluster_i[0], cluster_i[1]])
            elif len_clster <= 3:
                for j in range(len_clster):
                    self.edges_per_img.append([center1_i, cluster_i[j]])
            else:  # points num > 3:
                clusters2, centers2, loss2 = kmeans.get_cluster_custom \
                    (cluster_i, visualize=False, class_num=3)
                self.connect_nearest_to_mid(clusters2, centers2)
                self.recursive_get_edges(clusters2, centers2)

def get_and_line_edges(el,temp):
    clusters1, centers1, loss = kmeans.get_cluster_custom(el, visualize=False, class_num=10)
    print('initial-kmeans loss', loss)
    KmeansEdges = Gen_edges_with_kmeans_recursively()
    KmeansEdges.recursive_get_edges(clusters1, centers1)
    edges = KmeansEdges.get_final_edges()
    # connect the edges
    for j in range(len(edges)):
        temp = connect_points(temp, edges)

def project_L0_box(delta_x, k,k2,i,epoch_switch,USE_MULTI_PROCESS,device):
    p1 = torch.sum(torch.pow(delta_x,2), dim = [1])
    p3 = torch.sort(torch.reshape(p1, [p1.shape[0], -1]))[0]
    if i < epoch_switch:
        p3 = p3[:, -k] #倒数第k大
        temp = torch.unsqueeze(p1.ge(p3.view([-1, 1, 1])), 1)
    else:
        diff_topK = k2
        p3 = p3[:, -diff_topK]
        temp = torch.unsqueeze(p1.ge(p3.view([-1, 1, 1])), 1)
        idx = []
        #按序号切成4份
        for temp_elem in temp:
            idx.append(torch.nonzero(temp_elem, as_tuple=False).cpu().detach().numpy())
        temp = temp.cpu() #对tensor有多次非批量赋值，在cpu内部操作更快

        #speed up loop by multi-processing
        if USE_MULTI_PROCESS:
            p_pool_edge1 = mp.Pool()
        batch_i = 0
        for elem in idx: #elem 100*4
            el = elem[:,1:]
            temp_i = temp[batch_i]
            if USE_MULTI_PROCESS:
                p_pool_edge1.apply_async(get_and_line_edges, args=(el, temp_i))
            else:
                get_and_line_edges(el, temp_i)
            print('start multi-processing connection')

            batch_i += 1
        if USE_MULTI_PROCESS:
            p_pool_edge1.close()
            p_pool_edge1.join()
            p_pool_edge1.close()
        print(torch.sum(temp.float(), dim=[1, 2, 3]))

    temp = temp.to(device)
    return temp

def thres_loss_other03(y_pred):
    class_probability = y_pred[:,:-1]
    logit_threshold = 0.25
    p_max = class_probability.max(1)
    P_ge = p_max[0].ge(logit_threshold).float()*p_max[0]
    return torch.sum(P_ge)

from skimage import measure
def get_connected_domain_num(input_img,device):
    input_img = input_img.permute(2,0,1)
    ones = torch.ones_like(input_img[0]).to(device)
    zeros = torch.zeros_like(input_img[0]).to(device)

    input_img_tmp2 = torch.where((input_img[0] != 0), ones, zeros) + \
                     torch.where((input_img[1] != 0), ones, zeros) + \
                     torch.where((input_img[2] != 0), ones, zeros)
    input_map_new = torch.where(input_img_tmp2 > 0, ones, zeros)
    diff_points_num = torch.nonzero(input_map_new,as_tuple=False).shape
    return diff_points_num[0], measure.label(input_map_new.cpu().numpy()[:, :], background=0, connectivity=2)

def project_diff_points_to_ori_img(diff_t,img_ori,device,temp_mask):

    img_modify = img_ori.detach().clone()
    mask_idxs = torch.nonzero(temp_mask, as_tuple=False)

    for i in range(mask_idxs.shape[0]):
        idx = list(mask_idxs[i])
        for j in range(3):
            elem = diff_t[idx]
            if pixel_coordinates_checker(idx[0],idx[1]) == False: # 避免异常
                continue
            if elem >= 0 and elem < 1:
                if img_ori[idx] < 255:  # 避免将如0.1整型化为0，导致连通域被破坏
                    diff_t[idx] = 1
                else:
                    diff_t[idx] = -1
            if elem < 0 and elem > -1:
                if img_ori[idx] > 0:  # 避免将如-0.1整型化为0，导致连通域被破坏
                    diff_t[idx] = -1
                else:
                    diff_t[idx] = 1

            if torch.round(diff_t[idx]) == 0:
                print('err')
            img_modify[idx] = torch.round(img_ori[idx] + diff_t[idx])
            if img_modify[idx] > 255:
                if img_ori[idx] != 255:
                    img_modify[idx] = 255  # keep connected domain.
                else:
                    img_modify[idx] = 254  # keep connected domain.
            if img_modify[idx] < 0:
                if img_ori[idx] != 0:
                    img_modify[idx] = 0  # keep connected domain.
                else:
                    img_modify[idx] = 1  # keep connected domain.
            idx[2] += 1 # 3 ch
    x = torch.cat((temp_mask,temp_mask,temp_mask),dim=2)
    points1, labels1 = get_connected_domain_num(x, device)
    labels1 = labels1.reshape(500,500,1)
    cv2.imwrite('ori_mask.png',np.concatenate((labels1,labels1,labels1),axis=-1))
    print('mask 点数', points1, 'mask连通域：', np.max(labels1))
    #比较修改后与修改前的图片
    points,labels = get_connected_domain_num(img_modify-img_ori,device)
    labels = labels.reshape(500, 500, 1)
    cv2.imwrite('blur_mask.png', np.concatenate((labels, labels, labels), axis=-1))
    print('diff_modify 点数', points,'连通域：',np.max(labels))

    return img_modify

def save_atk_img(filename,diff_t,dir_path,device,temp_mask):
    # transpose channels to (w,h,ch)
    diff_t = diff_t.permute(1, 2, 0)
    temp_mask = temp_mask.permute(1, 2, 0)

    # 1. convert img to original 0~255
    # read ori data
    ori_img_path = '../select1000_new/' + filename
    img_ori = Image.open(ori_img_path).convert('RGB')
    img_ori_np = np.array(img_ori)
    img_ori = torch.tensor(img_ori_np, dtype=torch.float).to(device)
    # 2. project atk pixels' coordinate into original picture
    img_modify = project_diff_points_to_ori_img(diff_t,img_ori,device,temp_mask)

    #3. save
    file_save_path = dir_path + filename
    img_modify_np = img_modify.cpu().detach().clone().numpy()

    img_saved = Image.fromarray(img_modify_np.astype('uint8')).convert('RGB')
    try:
        img_saved.save(file_save_path)
    except:
        os.mkdir(dir_path)
        img_saved.save(file_save_path)
    print('3. save ok')
    return

from tool.darknet2pytorch import *

def load_mmdet_model(device):
    config = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = '../models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    model = init_detector(config, checkpoint, device=device)
    return model

def load_yoloV4_model(device):
    cfgfile = "../models/yolov4.cfg"
    weightfile = "../models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model = darknet_model.eval().to(device)
    darknet_model.load_weights(weightfile)
    return darknet_model

def load_img_preprocessing(filename):
    img = Image.open(filename).convert('RGB')
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    return img.float()

def load_img_batch(img_dir,batchsize,i,files_list,device):

    img_file_list = []
    data_batch = []
    # load data
    for img_child_idx in range(batchsize):
        try:
            file_name = files_list[i + img_child_idx]
            file_name = img_dir + '/' + file_name
            if os.path.splitext(file_name)[1] not in ['.jpg', '.png', '.bmp', '.gif']:
                continue
            data_batch.append(load_img_preprocessing(file_name))
            img_file_list.append(file_name)
        except:
            print('out of length')
    if len(data_batch) > 1:
        data_batch = torch.cat(data_batch, dim=0)  # batch_size * 3 * w * h
    else:
        data_batch = data_batch[0]
    return img_file_list, data_batch.to(device)

def get_loss_bg_yolo(y_pred):
    loss = 0
    idx = [4,4+85,4+85+85] #for 3 anchors
    nums = 0
    for i in range(len(y_pred)):
        temp_y = y_pred[i]
        temp_bg_score = torch.sigmoid(temp_y[:,idx])
        # if use thresh loss
        logit_threshold = 0.15
        P_ge = temp_bg_score.ge(logit_threshold).float()
        temp_bg_score = P_ge * temp_bg_score

        shape = temp_bg_score.shape
        nums += shape[0]*shape[1]*shape[2]*shape[3]
        loss += torch.sum(temp_bg_score)
    return loss/nums

def get_loss_bg_rcnn_stg1(y_pred):
    logit_threshold = 0.05
    P_ge = y_pred.ge(logit_threshold).float()
    temp_bg_score = P_ge * y_pred
    return torch.sum(temp_bg_score) / y_pred.shape[0]

def norm_img(x_atk2,device):
    # normalizer
    mean = torch.tensor([123.675, 116.28, 103.53]).reshape(1, -1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).reshape(1, -1).to(device)
    std_divisor = 1.0 / std

    x_atk = x_atk2.permute(0, 2, 3, 1)
    x_atk = (x_atk - mean) * std_divisor
    x_atk = x_atk.permute(0, 3, 1, 2)
    return x_atk

def lower_train_test_difference_by_retraining(diff_atk,temp_mask,x):
    diff_atk = torch.round(diff_atk)
    # 对于第一通道，如果temp_mask为True而diff_atk==0
    temp_mask_problem = torch.logical_xor(temp_mask, diff_atk[:, :1, :, :])
    problem_elems = torch.nonzero(temp_mask_problem, as_tuple=False)
    # 将diff_atk修改为1或-1
    for elem in problem_elems:
        idx = list(elem)
        if x[idx] == 255:
            diff_atk[idx] = -1
        else:
            diff_atk[idx] = 1
    return diff_atk

def one_process_train():

    img_dir = '../select1000_new/'
    files_list = os.listdir(img_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_yolo = load_yoloV4_model(device=device)
    model_rcnn = load_mmdet_model(device)

    batchsize = 2
    save_dir_path = '../select1000_new_p/'
    sparsity_K = 8000
    max_epoch = 250
    quantize_epoch = 30
    epoch_switch = 30
    top_K2 = 200 #this needs to be fine-tuned for different img
    lr = 120000.0 / 2

    USE_MOMENTUM = True
    USE_MULTI_PROCESS = False

    upsample_yolo = torch.nn.UpsamplingBilinear2d(size=(608,608))
    upsample_rcnn = torch.nn.UpsamplingBilinear2d(size=(800,800))

    for i in tqdm(range(0,len(files_list), batchsize)):

        img_file_list, x = load_img_batch(img_dir,batchsize, i, files_list,device)
        data_rcnn_atk = data_loader_mmdet(model_rcnn, img_file_list)

        x_atk = x.clone().detach().to(device)
        current_bs = x.shape[0]

        grad_momentum = torch.zeros_like(x_atk).to(device)
        for ep in range(max_epoch+quantize_epoch+1):
            x_atk.requires_grad_()
            x_atk.retain_grad()
            #part yolo
            x_atk_yolo = torch.div(upsample_yolo(x_atk),255)
            x_atk_yolo.requires_grad_()
            y_pred1 = model_yolo(x_atk_yolo)
            #part rcnn
            x_atk_rcnn = norm_img(upsample_rcnn(x_atk),device)
            x_atk_rcnn.requires_grad_()
            data_rcnn_atk['img'][0] = x_atk_rcnn
            stg2,stg1 = inference_detector_with_grad(model_rcnn, data_rcnn_atk)

            loss1 = get_loss_bg_yolo(y_pred1)
            loss_stg1 = get_loss_bg_rcnn_stg1(stg1)
            loss_stg2 = thres_loss_other03(stg2)/ stg2.shape[0]

            # lower the score of background
            loss = loss1 + (loss_stg1 / 10 + loss_stg2 / 60)/5
            model_yolo.zero_grad()
            model_rcnn.zero_grad()
            loss.backward()

            if ep % 5 == 0:
                print()
                print('epoch:{:.0f} loss: {:.4f} yolo: {:.4f} stg1: {:.4f} stg2: {:.4f}'
                      .format(ep,loss.data*1000, loss1.data*1000,loss_stg1.data*100,loss_stg2.data*1000/60 ))
            with torch.no_grad():
                grad = x_atk.grad
                #grad += grad2
                grad /= (1e-10 + torch.sum(torch.abs(grad), dim=[1, 2, 3], keepdim=True))
                # gradient descent
                rd_noise = ((torch.rand(grad.shape) - 0.5) * 1e-12).to(device)
                if USE_MOMENTUM:
                    grad_momentum = 0.5 * grad_momentum + grad
                    x_atk -= rd_noise + lr * grad_momentum
                else:
                    x_atk -= rd_noise + lr * grad

                x_atk= x_atk.clamp(0,255)
                diff_atk = x_atk - x
                if ep <= epoch_switch: #投影
                    temp_mask = project_L0_box(diff_atk, sparsity_K, top_K2,ep,epoch_switch,USE_MULTI_PROCESS,device)
                diff_atk = temp_mask * diff_atk
                if ep > max_epoch and ep % 4 == 0:
                    diff_atk = lower_train_test_difference_by_retraining(diff_atk,temp_mask,x)
                x_atk = x + diff_atk

        if USE_MULTI_PROCESS:
            p_pool_save = mp.Pool()
        for i_save in range(current_bs):
            file_name = files_list[i + i_save]
            if USE_MULTI_PROCESS:
                p_pool_save.apply_async(save_atk_img,args=(file_name, diff_atk[i_save], save_dir_path, device,temp_mask[i_save]))
            else:
                save_atk_img(file_name, diff_atk[i_save], save_dir_path, device,temp_mask[i_save])
        if USE_MULTI_PROCESS:
            p_pool_save.close()
            p_pool_save.join()
            p_pool_save.close()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    one_process_train()

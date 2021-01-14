import sys
import UI_util
import numpy as np
import cv2
import time
import math
from xai_viewer_ui import Ui_xai_viewer

# sys.path.append('../')
# sys.path.append('../nodule_detector')
# sys.path.append('../nodule_classifier')

import torch
# import training.res18_split_focal as detect_model
# import training.res_classifier as attribute_model
import res18_split_focal as detect_model
import res_classifier as attribute_model
import rule
from torch.nn import DataParallel
from torch.backends import cudnn
# from training.utils import *
# from training.split_combine import SplitComb
import SimpleITK as sitk
from utils import *
from split_combine import SplitComb
from save_sitk import savedicom
#TODO: nodule view rescale feature add
class savefile():
    def __init__(self, filename):
        super(savefile,self).__init__()
        #self.init_openpath = '/root/ssd_data/LUNA/allset/'
        self.init_openpath = '/home/imsight/data/demo/'
        #self.init_openpath = '/root/ssd_data/LUNA/allset/'
        self.resolution = np.array([1,1,1])
        self.slice_index = 0
        self.slice_num = 0
        self.slice_width = 0
        self.slice_height = 0
        self.detect_resume = './detector.ckpt'
        self.attribute_resume = './attribute.ckpt'
        self.gpu = '0'
        self.detect_net, self.attribute_net, self.split_comber, self.get_pbb \
            = self.init_net()
        self.stride = 4
        self.n_per_run = 1
        self.rule_data_dir = './data/'
        self.nodule_rule_prob_count(self.rule_data_dir)
        self.filename = filename
        self.open(self.filename)
        self.detect()

    def nodule_rule_prob_count(self, rule_data_dir):
        labels = []
        nodule_labels = []

        # list for count,
        # 7attribute(calcification, lobulation, margin, sphericiy, spiculation, subtlety, texture)
        # 3 malignamcy(benign, neutral, malignant)
        mal_rule_count_list = np.zeros((7, 3))
        neutral_rule_count_list = np.zeros((7, 3))
        benign_rule_count_list = np.zeros((7, 3))

        mal_rule_prob = np.zeros((7, 3))
        neutral_rule_prob = np.zeros((7, 3))
        benign_rule_prob = np.zeros((7, 3))

        # rule count attribute key
        self.rule_count_name = ['calcification', 'lobulation', 'margin', 'sphericity',
                           'spiculation', 'subtlety', 'texture']

        # attribute rule explain word
        mal_attr_dict = {'calcification': 'calcification is absent', 'lobulation': 'lobulation is none',
                         'margin': 'margin is Poorly defined',
                         'sphericity': 'sphericity is Linear', 'spiculation': 'spiculation is marked',
                         'subtlety': 'subtlety is obvious', 'texture': 'texture is solid',
                         'medium': 'size is medium', 'small': 'size is small', 'big': 'size is big'}
        neutral_attr_dict = {'calcification': '', 'lobulation': 'lobulation is meidum', 'margin': 'margin is meidum',
                             'sphericity': 'sphericity is ovoid', 'spiculation': 'spiculation is meidum',
                             'subtlety': 'subtlety is subtle', 'texture': 'texture is part solid',
                             'medium': 'size is medium', 'small': 'size is small', 'big': 'size is big'}
        benign_attr_dict = {'calcification': '', 'lobulation': 'lobulation is marked',
                            'margin': 'margin is poorly sharp',
                            'sphericity': 'sphericity is Round', 'spiculation': 'spiculation is marked',
                            'subtlety': 'subtlety is subtle', 'texture': 'texture is nonsolid',
                            'medium': 'size is medium', 'small': 'size is small', 'big': 'size is big'}

        self.attr_dict = [benign_attr_dict, neutral_attr_dict, mal_attr_dict]

        # list for size count
        # 3 rule (small, medium, bing),
        # 3 malignamcy(benign, neutral, malignant)a
        size_rule_count_list = np.zeros((3, 3))
        # list for size prob
        self.size_rule_prob = np.zeros((3, 3))

        self.size_rule_count_name = ['small', 'medium', 'big']

        # load attribute
        for idx in range(1):
            # l = np.load(os.path.join(data_dir, '%s_attribute.npy' % idx))

            num = str(idx)
            for i in range(3 - len(str(idx))):
                num = str('0') + num

            path = rule_data_dir + num + '_attribute.npy'
            l = np.load(path)

            if np.all(l == 0):
                l = np.array([])
            labels.append(l)

        for i, l in enumerate(labels):
            if len(l) > 0:
                for label in l:
                    nodule_labels.append([np.concatenate([[i], label[1:]])])

        nodule_labels = np.concatenate(nodule_labels, axis=0)

        print(np.shape(nodule_labels))

        # attribute score rounding
        for i in range(np.shape(nodule_labels)[0]):
            nodule_labels[i][4] = round(nodule_labels[i][4], 2)
            nodule_labels[i][5] = round(nodule_labels[i][5], 2)
            nodule_labels[i][6] = round(nodule_labels[i][6], 2)
            nodule_labels[i][7] = round(nodule_labels[i][7], 2)
            nodule_labels[i][8] = round(nodule_labels[i][8], 2)
            nodule_labels[i][9] = round(nodule_labels[i][9], 2)
            nodule_labels[i][10] = round(nodule_labels[i][10], 2)
            nodule_labels[i][11] = round(nodule_labels[i][11], 2)
            nodule_labels[i][12] = round(nodule_labels[i][12], 2)
            nodule_labels[i][13] = round(nodule_labels[i][13], 2)

        # count benign, neutral, malignant per rule
        for i in range(np.shape(nodule_labels)[0]):
            # print_label(nodule_labels[i])
            rule.mal_count_rule(nodule_labels[i], mal_rule_count_list)
            rule.neutral_count_rule(nodule_labels[i], neutral_rule_count_list)
            rule.benign_count_rule(nodule_labels[i], benign_rule_count_list)
            rule.size_count_rule(nodule_labels[i], size_rule_count_list)

        # calculate benign, neutral, malignancy prob
        rule.calc_prob(mal_rule_count_list, neutral_rule_count_list, benign_rule_count_list, self.rule_count_name,
                  mal_rule_prob, neutral_rule_prob, benign_rule_prob,
                  size_rule_count_list, self.size_rule_count_name, self.size_rule_prob)

        self.prob_list = [benign_rule_prob, neutral_rule_prob, mal_rule_prob]
        print(self.size_rule_prob)
        print(benign_rule_prob)
        print(neutral_rule_prob)
        print(mal_rule_prob)

        # # print all noudle sentence
        # for i in range(np.shape(nodule_labels)[0]):
        #     rule(nodule_labels[i], prob_list, size_rule_prob, rule_count_name, size_rule_count_name, attr_dict)

    def init_net(self):
        torch.manual_seed(0)
        torch.cuda.set_device(0)

        #model = import_module(self.model)
        detect_config, detect_net, _, get_pbb = detect_model.get_model()
        attribute_config, attribute_net, __ = attribute_model.get_model()

        detect_checkpoint = torch.load(self.detect_resume)
        detect_net.load_state_dict(detect_checkpoint['state_dict'])

        attribute_checkpoint = torch.load(self.attribute_resume)
        attribute_net.load_state_dict(attribute_checkpoint['state_dict'])


        n_gpu = setgpu(self.gpu)

        detect_net = detect_net.cuda()
        attribute_net = attribute_net.cuda()
        #loss = loss.cuda()
        cudnn.benchmark = True
        detect_net = DataParallel(detect_net)
        attribute_net = DataParallel(attribute_net)

        margin = 32
        sidelen = 144
        split_comber = SplitComb(sidelen, detect_config['max_stride'], detect_config['stride'], margin, detect_config['pad_value'])

        print ("init_net complete")
        return detect_net, attribute_net, split_comber, get_pbb

    def rule(self, label, prob_list, size_rule_prob, rule_count_name, size_rule_count_name, attr_dict):
        size = label[2]
        mal = label[0]
        sphericity = label[3]
        margin = label[4]
        spiculation = label[5]
        texture = label[6]
        calcification = label[7]
        internal_structure = label[8]
        lobulation = label[9]
        subtlety = label[10]

        print("#############################################")
        mal_res = rule.check_mal(mal)

        #### check attribute has rule
        rule_idx = []
        size_idx = 0

        # benign rule check
        sentence = ' '
        if (mal_res == 0):
            sentence = "nodule is benign"
            #print("mal is benign")
            if (lobulation <= 3):
                rule_idx.append(1)
            if (margin >= 3):
                rule_idx.append(2)
            if (sphericity >= 3):
                rule_idx.append(3)
            if (spiculation <= 2):
                rule_idx.append(4)
            if (subtlety <= 3):
                rule_idx.append(5)
            if (texture <= 3):
                rule_idx.append(6)
        # neutral rule check
        elif (mal_res == 1):
            sentence = "nodule is neutral"
            #print("mal is neutral")
            if (3 < lobulation and lobulation <= 4):
                rule_idx.append(1)
            if (3 > margin and lobulation >= 2):
                rule_idx.append(2)
            if (3 > sphericity and lobulation >= 2):
                rule_idx.append(3)
            if (2 < spiculation and lobulation <= 3):
                rule_idx.append(4)
            if (3 < subtlety and lobulation <= 4):
                rule_idx.append(5)
            if (3 < texture and lobulation <= 4):
                rule_idx.append(6)
        # malignant rule check
        elif (mal_res == 2):
            sentence = "nodule is malignant"
            #print("mal is malignant")
            if (calcification == 6):
                rule_idx.append(0)
            if (lobulation > 4):
                rule_idx.append(1)
            if (margin < 2):
                rule_idx.append(2)
            if (sphericity < 2):
                rule_idx.append(3)
            if (spiculation > 3):
                rule_idx.append(4)
            if (subtlety > 4):
                rule_idx.append(5)
            if (texture > 4):
                rule_idx.append(6)

        # check size rule
        if (size < 5):
            size_idx = 0
        if (5 <= size and size < 10):
            size_idx = 1
        if (10 < size):
            size_idx = 2

        prob_dict = {}

        # get size rule prob
        prob_dict[size_rule_count_name[size_idx]] = size_rule_prob[size_idx][mal_res]

        # get rule prob
        for i in range(len(rule_idx)):
            prob_dict[rule_count_name[rule_idx[i]]] = prob_list[mal_res][rule_idx[i]][mal_res]

        # sorting rule prob
        prob_dict_sort = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)

        # print top 3 rule sentence
        top_count = 0
        sentence = sentence + ' because '
        #print('because', end=' ')
        for key, value in prob_dict_sort:
            top_count = top_count + 1
            sentence = sentence + attr_dict[mal_res][key] + ' '
            #print(attr_dict[mal_res][key], end=' ')

            if (key == 'medium' or key == 'small' or key == 'big'):
                sentence = sentence + '(' + str(size) + 'mm)' + ' '
                #print('(' + str(size) + 'mm)', end=' ')
            if (top_count >= 3):
                break
            else:
                sentence = sentence + 'and' + ' '
                #print('and', end=' ')
        #print()
        return sentence


    def print_nodule_attribute(self, attrbute_list):


        hori_list_item = ['malignancy', 'nodule_prob', 'size',
                          'sphericiy', 'margin', 'spiculation',
                          'texture', 'calcification', 'internal_structure',
                          'lobulation', 'subtlety']

        num = 0
        nodule_labels_all=[]
        sentence_all=[]
        for i in range(len(self.world_pbb)):


            nodule_labels = []
            nodule_labels.append(round(attrbute_list[i][0][0], 2))
            nodule_labels.append(round(self.world_pbb[i][0] * 100, 2))
            nodule_labels.append(round(self.world_pbb[i][4], 2))
            nodule_labels.append(round(attrbute_list[i][0][1], 2))
            nodule_labels.append(round(attrbute_list[i][0][2], 2))
            nodule_labels.append(round(attrbute_list[i][0][3], 2))
            nodule_labels.append(round(attrbute_list[i][0][4], 2))
            nodule_labels.append(round(attrbute_list[i][0][5], 2))
            nodule_labels.append(round(attrbute_list[i][0][6], 2))
            nodule_labels.append(round(attrbute_list[i][0][7], 2))
            nodule_labels.append(round(attrbute_list[i][0][8], 2))
            sentence = self.rule(nodule_labels, self.prob_list, self.size_rule_prob, self.rule_count_name,
                                 self.size_rule_count_name, self.attr_dict)
            print(sentence)
            # JZWANG SAVE
            sentence_all.append(sentence)
            nodule_labels_all.append(nodule_labels)
        labels_filename = "result/labels_"+self.pt_num + ".npy"
        np.save(labels_filename, nodule_labels_all)
        txtfilename = "result/sentence_"+self.pt_num+".txt"
        with open(txtfilename,"w") as f:
            f.write("\n".join(sentence_all))



    def detect(self):

        if (self.slice_num <= 0):
            return 0

        s = time.time()
        self.gt_path = '/research/dept8/jzwang/code/lung_nodule_integ_viewer/data/' + self.pt_num + '_label.npy'
        data, coord2, nzhw = UI_util.split_data(np.expand_dims(self.sliceim_re, axis=0),
                                                self.stride, self.split_comber)


        labels = np.load(self.gt_path)

        e = time.time()

        self.lbb, self.world_pbb = UI_util.predict_nodule_v1(self.detect_net, data, coord2, nzhw, labels,
                               self.n_per_run, self.split_comber, self.get_pbb)

        nodule_items = []
        for i in range(len(self.lbb)):
            if self.lbb[i][3] != 0:
                nodule_items.append('gt_' + str(i))

        for i in range(len(self.world_pbb)):
            nodule_items.append('cand_' + str(i) + ' ' + str(round(self.world_pbb[i][0], 2)))


        print('elapsed time is %3.2f seconds' % (e - s))
        UI_util.draw_nodule_rect(self.lbb, self.world_pbb, self.slice_arr)
        num = 0
        print(" self.slice_arr.shape:", self.slice_arr.shape)
        attrbute_list = []
        labels_filename = "result/world_pbb_"+self.pt_num+".npy"
        np.save(labels_filename, self.world_pbb)
        for i in range(len(self.world_pbb)):
            print ("self.world_pbb:", self.world_pbb[i][1:])
            print (np.shape(self.sliceim_re))
            crop_img, _ = UI_util.crop_nodule_arr_2ch(self.world_pbb[i][1:], np.expand_dims(self.sliceim_re, axis=0))
            output = UI_util.predict_attribute(self.attribute_net, crop_img.unsqueeze(0))
            print (output.cpu().data.numpy())
            attrbute_list.append(output.cpu().data.numpy())


            #print ("/root/workspace/dsb2017_review/DSB2017_1/training/XAI_UI/test1" + str(i) + ".png")
            #print ("/root/workspace/dsb2017_review/DSB2017_1/training/XAI_UI/test2" + str(i) + ".png")
            #cv2.imwrite("/root/workspace/dsb2017_review/DSB2017_1/training/XAI_UI/test1" + str(i) + ".png", crop[0][24])
            #cv2.imwrite("/root/workspace/dsb2017_review/DSB2017_1/training/XAI_UI/test2" + str(i) + ".png", crop[1][24])
        self.print_nodule_attribute(attrbute_list)


    def open(self, file):
        #TODO: file type ch
        fileName = []
        fileName.append(file)
        print("open ",fileName)
        if (fileName[0] == ''):
            return 0

        self.pt_num = fileName[0].split('/')[-1].split('.mhd')[0]


        sliceim, origin, spacing, isflip = UI_util.load_itk_image(fileName[0])
        print("sliceim.shape", sliceim.shape)
        print("origin.shape", origin.shape)
        print("spacing.shape", spacing.shape)
        print(spacing)

        if isflip:
            sliceim = sliceim[:, ::-1, ::-1]
            print('flip!')
        sliceim = UI_util.lumTrans(sliceim)


        self.sliceim_re, _ = UI_util.resample_v1(sliceim, spacing, self.resolution, order=1)
        savedicom(outputdir="result/"+self.pt_num, input=self.sliceim_re,  pixel_dtypes="int16")

        self.slice_arr = np.zeros((np.shape(self.sliceim_re)[0], np.shape(self.sliceim_re)[1], np.shape(self.sliceim_re)[2], 3))

        self.slice_num = np.shape(self.sliceim_re)[0]
        self.slice_height = np.shape(self.sliceim_re)[1]
        self.slice_width = np.shape(self.sliceim_re)[2]

        for i in range(len(self.sliceim_re)):
            self.slice_arr[i] = cv2.cvtColor(self.sliceim_re[i], 8)

        print ("finish convert")
        self.slice_index = int(self.slice_num/2)
        img = np.array(self.slice_arr[self.slice_index], dtype=np.uint8)





if __name__ == '__main__':
    savefile(filename="/research/dept8/jzwang/code/lung_nodule_integ_viewer/data/001.mhd")

import xml.etree.ElementTree as ET
import pickle
import os
import re
import sys
import shutil
import math
from os import listdir, getcwd
from os.path import join
import cv2
import numpy as np

car = ["saloon_car", "suv", "van", "pickup"]
other = ["shop_truck", "unknown"]
bicycle = ["bicycle", "motorcycle"]

targettypes = ["car",
               "car_front",
               "car_rear",
               "bicycle",
               "person",
               "cyclist",
               "tricycle",
               "motorcycle",
               "non_interest_zone",
               "non_interest_zones"]
classes_c9 = ["car",
              "truck",
              "waggon",
              "passenger_car",
              "other",
              "bicycle",
              "person",
              "cyclist",
              "tricycle",
              "non_interest_zone"]
classes_c6 = ['car',
              "bicycle",
              "person",
              "cyclist",
              "tricycle",
              "car_fr",
              "non_interest_zone",
              "non_interest_zones"]
classes_c5 = ['car',                 # 0
              "bicycle",             # 1
              "person",              # 2
              "cyclist",             # 3
              "tricycle",            # 4
              "non_interest_zone"]

# classes = classes_c6
classes = classes_c5
class_num = len(classes) - 1  # 减1减的是non_interest_zone
car_fr = ["car_front", "car_rear"]

nCount = 0


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    xmin = box[0]
    xmax = box[1]
    ymin = box[2]
    ymax = box[3]
    if xmin < 0:
        xmin = 0
    if xmax < 0 or xmin >= size[0]:
        return None
    if xmax >= size[0]:
        xmax = size[0] - 1
    if ymin < 0:
        ymin = 0
    if ymax < 0 or ymin >= size[1]:
        return None
    if ymax >= size[1]:
        ymax = size[1] - 1
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = abs(xmax - xmin)
    h = abs(ymax - ymin)
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    if w == 0 or h == 0:
        return None
    return (x, y, w, h)


def convert_annotation(imgpath, xmlpath, labelpath, filename):
    in_file = open(xmlpath+'/'+filename+'.xml')
    out_file = open(labelpath+'/'+filename+'.txt', 'w')
    xml_info = in_file.read()

    if xml_info.find('dataroot') < 0:
        print("Can not find dataroot")
        out_file.close()
        in_file.close()
        return [], []

    #xml_info = xml_info.decode('GB2312').encode('utf-8')
    #xml_info = xml_info.replace('GB2312', 'utf-8')

    try:
        root = ET.fromstring(xml_info)
    except(Exception, e):
        print("Error: cannot parse file")
        #n = raw_input()
        out_file.close()
        in_file.close()
        return [], []

    boxes_non = []
    poly_non = []
    # Count = 0
    label_statis = [0 for i in range(class_num)]
    if root.find('markNode') != None:
        obj = root.find('markNode').find('object')
        if obj != None:
            w = int(root.find('width').text)
            h = int(root.find('height').text)
            print("w:%d, h%d" % (w, h))
            # print 'w=' + str(w) + ' h=' + str(h)
            for obj in root.iter('object'):
                target_type = obj.find('targettype')
                cls_name = target_type.text
                print(cls_name)
                if cls_name not in targettypes:
                    print("********************************* "+cls_name +
                          " is not in targetTypes list *************************")
                    continue
                # # classes_c9

                # if cls_name == "car":
                #     cartype = obj.find('cartype').text
                #     # print(cartype)
                #     if cartype == 'motorcycle':
                #         cls_name = "bicycle"
                #     elif cartype == 'truck':
                #         cls_name = "truck"
                #     elif cartype == 'waggon':
                #         cls_name = 'waggon'
                #     elif cartype == 'passenger_car':
                #         cls_name = 'passenger_car'
                #     elif cartype == 'unkonwn' or cartype == "shop_truck":
                #         cls_name = "other"

                # classes_c5
                if cls_name == 'car_front' or cls_name == 'car_rear':
                    cls_name = 'car_fr'
                if cls_name == 'car':
                    cartype = obj.find('cartype').text
                    if cartype == 'motorcycle':
                        cls_name = 'bicycle'
                if cls_name == "motorcycle":
                    cls_name = "bicycle"
                if cls_name not in classes:
                    print("********************************* " + cls_name +
                          " is not in class list *************************")
                    continue

                cls_id = classes.index(cls_name)
                # print(cls,cls_id)
                cls_no = cls_id
                # elif 'non_interest_zone' == cls:
                #     imgfile = imgpath + '/'+filename+'.jpg'
                #     img = cv2.imread(imgfile)
                #     xmin = int(xmlbox.find('xmin').text)
                #     xmax = int(xmlbox.find('xmax').text)
                #     ymin = int(xmlbox.find('ymin').text)
                #     ymax = int(xmlbox.find('ymax').text)
                #     print(xmin,xmax,ymin,ymax,img.shape)
                #     tmp = np.zeros((ymax-ymin,xmax-xmin,3),img.dtype)
                #     img[ymin:ymax,xmin:xmax] = tmp
                #     cv2.imwrite(imgfile,img)
                #     print("has non_interest_zone*************************************************************")
                #     continue

                # print(cls_no)
                # if cls_no != 6:
                #    continue

                if cls_name == "non_interest_zones":  # 有个bug, non_interest_zones时为bndbox,胡老板已修复。
                    try:
                        xmlpoly = obj.find('polygonPoints').text
                        print('xml_poly:', xmlpoly)
                        poly_ = re.split('[,;]', xmlpoly)
                        poly_non.append(poly_)
                        continue
                    except:
                        continue

                # Count += 1
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text),
                     float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))

                if cls_name == "non_interest_zone":
                    boxes_non.append(b)
                    continue

                #
                label_statis[cls_no] += 1
                bb = convert((w, h), b)
                if bb is None:
                    print("++++++++++++++++++++++++++++++box is error++++++++++++++++++++")
                    # sleep(10)
                    continue
                
                out_file.write(str(cls_no) + " " +
                               " ".join([str(a) for a in bb]) + '\n')
                print(str(cls_no) + " " + " ".join([str(a) for a in bb]))

    out_file.close()
    in_file.close()

    # if Count > 0:
    #     return 0
    # else:
    #     # if os.path.exists(labelpath+'/'+filename+'.txt'):
    #     #    os.remove(labelpath+'/'+filename+'.txt')
    #     return -1
    return poly_non, boxes_non, label_statis


if __name__ == "__main__":

    # rootdir = '/users/maqiao/mq/Data_checked/multiClass/multiClass0320'
    # root_path = "/users/maqiao/mq/Data_checked/multiClass/pucheng20191101"
    # rootdirs = [
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass0320',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass0507',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass0606',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass0704',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190808',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190814',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190822-1',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190822-3',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190823',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190826',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190827',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190827_1',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190830',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190830_1',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190830_2',
                # '/users/maqiao/mq/Data_checked/multiClass/multiClass190830_3'
                # "/users/maqiao/mq/Data_checked/multiClass/mark/houhaicui",
                # "/users/maqiao/mq/Data_checked/multiClass/mark/limingqing",
                # "/users/maqiao/mq/Data_checked/multiClass/mark/mayanzhuo",
                # "/users/maqiao/mq/Data_checked/multiClass/mark/quanqingfang",
                # "/users/maqiao/mq/Data_checked/multiClass/mark/shenjinyan",
                # "/users/maqiao/mq/Data_checked/multiClass/mark/wanglinan",
                # "/users/maqiao/mq/Data_checked/multiClass/mark/yangyanping",
                # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/houhaicui",
                # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/limingqing",
                # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/mayanzhuo",
                # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/quanqingfang",
                # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/shenjinyan",
                # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/wanglinan",
                # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/yangyanping",
                # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20190912",
                # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20190920",
                # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20190925",
                # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20190930",
                # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20191011",
                # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20191018",
                # "/users/maqiao/mq/Data_checked/multiClass/pucheng20191012",
                # "/users/maqiao/mq/Data_checked/multiClass/pucheng20191017",
                # "/users/maqiao/mq/Data_checked/multiClass/pucheng20191025",
                # "/users/maqiao/mq/Data_checked/multiClass/pucheng20191101"]
                # changsha_test_poly_nointer
                # /mnt/diskb/maqiao/multiClass/beijing20200110
                # /mnt/diskb/maqiao/multiClass/changsha20191224-2

    root_path = '/mnt/diskb/maqiao/multiClass/c5_puer_20200611'
    rootdirs = ["/mnt/diskb/maqiao/multiClass/c5_puer_20200611"]

    # root_path = '/users/duanyou/backup_c5/changsha_c5/test_new_chuiting'
    # rootdirs =  ["/users/duanyou/backup_c5/changsha_c5/test_new_chuiting"]

    # root_path = 'F:/mq1/test_data'
    # rootdirs  = [root_path+'/1']

    all_list_file = os.path.join(root_path, 'multiClass_train.txt')
    all_list = open(os.path.join(root_path, all_list_file), 'w')
    dir_num = len(rootdirs)
    for j, rootdir in enumerate(rootdirs):
        imgpath = rootdir + '/' + "JPEGImages_ori"
        imgpath_dst = rootdir + '/' + "JPEGImages"
        xmlpath = rootdir + '/' + "Annotations"
        labelpath = rootdir + '/' + "labels"
        if not os.path.exists(labelpath):
            os.makedirs(labelpath)
        if not os.path.exists(imgpath_dst):
            os.makedirs(imgpath_dst)

        list_file = open(rootdir + '/' + "train.txt", 'w')
        file_lists = os.listdir(imgpath)
        file_num = len(file_lists)

        label_count = [0 for i in range(class_num)]
        for i, imgname in enumerate(file_lists):
            print("**************************************************************************************" +
                  str(i) + '/' + str(file_num)+'  ' + str(j) + '/' + str(dir_num))
            print(imgpath + '/' + imgname)
            print(xmlpath+'/' + imgname[:-4] + ".xml")

            if imgname.endswith('.jpg') and os.path.exists(xmlpath+'/'+imgname[:-4]+".xml"):
                if not os.path.exists(imgpath):  # 没有对应的图片则跳过
                    continue

                poly_non, boxes_non, label_statis = convert_annotation(
                    imgpath, xmlpath, labelpath, imgname[:-4])
                print('boxes_on:', boxes_non)
                if label_statis == []:
                    continue
                label_count = [label_count[i] + label_statis[i]
                               for i in range(class_num)]

                img_ori = imgpath + '/' + imgname
                img = cv2.imread(img_ori)
                if img is None:
                    continue

                # 把不感兴趣区域替换成颜色随机的图像块
                is_data_ok = True
                if len(boxes_non) > 0:
                    for b in boxes_non:
                        xmin = int(min(b[0], b[1]))
                        xmax = int(max(b[0], b[1]))
                        ymin = int(min(b[2], b[3]))
                        ymax = int(max(b[2], b[3]))
                        if xmax > img.shape[1] or ymax > img.shape[0]:
                            is_data_ok = False
                            break
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if xmax > img.shape[1] - 1:
                            xmax = img.shape[1] - 1
                        if ymax > img.shape[0] - 1:
                            ymax = img.shape[0] - 1
                        h = int(ymax - ymin)
                        w = int(xmax - xmin)
                        img[ymin:ymax, xmin:xmax, :] = np.random.randint(
                            0, 255, (h, w, 3))  # 替换为马赛克

                # 把不感兴趣多边形区域替换成黑色
                if len(poly_non) > 0:
                    for poly in poly_non:
                        arr = []
                        i = 0

                        while i < len(poly) - 1:
                            arr.append([int(poly[i]), int(poly[i + 1])])
                            i = i + 2

                        arr = np.array(arr)
                        print('arr:', arr)
                        cv2.fillPoly(img, [arr], 0)

                if not is_data_ok:
                    continue

                img_dst = imgpath_dst + '/' + imgname
                print(img_dst)
                cv2.imwrite(img_dst, img)

                list_file.write(img_dst+'\n')
                all_list.write(img_dst+'\n')
            print("label_count ", label_count)

        list_file.close()
    all_list.close()

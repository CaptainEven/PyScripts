# coding=utf-8

import os
import sys

import xml.etree.ElementTree as ET
import shutil
import xlwt
import cv2
import pickle
from pylab import *
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, \
    QLineEdit, QLabel, QPushButton, QProgressBar, QRadioButton, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap


# ----------------------------------------------------
class Signal(QObject):
    signal = pyqtSignal()


class Analyser(object):
    def __init__(self, test_root, target_type):
        """
        :param test_root:
        :param target_type:
        """
        super().__init__()

        self.test_root = test_root
        self.target_type = target_type

        self.PR = []  # 运行一个PR的结果
        self.PRs = []  # 运行多个PR的结果
        self.pr_img_path = ''

        self.do_false_miss = False
        self.false_miss = []
        self.prob_th = 0.0

        # 游标
        self.cursor = 0
        self.pr_idx = 0
        self.cycle = 0

        # 事件信号
        self.cursor_signal = Signal()
        self.pr_idx_signal = Signal()
        self.ana_done_signal = Signal()

    def set_test_root(self, test_root):
        """
        :param test_root:
        :return:
        """
        self.test_root = test_root

        # ---------------- dirs
        if self.test_root != '':
            self.img_dir = test_root + '/' + 'JPEGImages'
            self.anno_dir = test_root + '/' + 'Annotations'
            self.det_dir = self.img_dir + '/' + 'detect_result'
            self.output_dir = '%s/show_result' % (self.det_dir)
            self.f_list_path = test_root + '/' + 'train.txt'
            if not (os.path.isdir(self.img_dir) and
                    os.path.isdir(self.anno_dir)):
                print('=> test set is invalid')
                print('=> img_dir: ', self.img_dir)
                print('=> anno_dir: ', self.anno_dir)
                return
            if not os.path.isfile(self.f_list_path):
                print('=> test set txt file is invalid.')
                return
            if not os.path.isdir(self.det_dir):
                print('=> detection result is invalid.')
                return
            print('=> dirs done.')

            # 加载样本名称列表
            self.f_list = self.load_f_list(self.f_list_path)
            print('=> Analyser file list loaded.')

        print('=> set test root: ', self.test_root)

    def load_f_list(self, f_path):
        """
        :param f_path:
        :return:
        """
        if not os.path.isfile(f_path):
            print('=> invalid file list.')
            return

        f_list = []
        with open(f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                f_name = os.path.split(line.strip())[-1][:-4]
                f_list.append(f_name)
        return f_list

    def load_result(self, result_path, fr):
        """
        读取检测结果
        :param result_path:
        :param fr:
        :return:
        """
        cn = 0
        detect_objs = []

        for line in fr.readlines():  # 依次读取每行
            line = line.strip()  # 去掉每行头尾空白
            if line.split()[0].isalpha():  # 是字母, 跳入下一行
                continue
            # if cn == 0:
            #     tmp, num = [str(i) for i in line.split("=")]
            else:
                obj = [float(i) for i in line.split()]
                obj[0] = int(obj[0])
                detect_objs.append(obj)
                # print(obj)
            cn += 1
        return detect_objs

    def load_label(self,
                   label_path,
                   f_label,
                   f_name,
                   target_type):
        """
        :param label_path:
        :param f_label:
        :param f_name:
        :param target_type:
        :return:
        """
        # read jpeg file
        jpg_path = str(label_path + '/' + f_name + '.jpg') \
            .replace('Annotations', 'JPEGImages')
        jpg_file = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
        img_w = jpg_file.shape[1]
        img_h = jpg_file.shape[0]

        cn = 0
        num = 0
        objs_gt = []
        label_info = f_label.read()
        if label_info.find('dataroot') < 0:
            print("[Err]: Can not find dataroot")
            f_label.close()
            return objs_gt

        try:
            root = ET.fromstring(label_info)
        except Exception as e:
            print("[Err]: cannot parse file")
            # n = raw_input()
            f_label.close()
            return objs_gt

        if root.find('markNode') != None:
            obj = root.find('markNode').find('object')
            if obj != None:
                if root.find('width') == None or root.find('height') == None:
                    w, h = img_w, img_h
                else:
                    w = int(root.find('width').text)
                    h = int(root.find('height').text)

                for obj in root.iter('object'):
                    cls = obj.find('targettype').text
                    if cls in target_type:
                        # class index modification...
                        cls_id = target_type.index(cls)
                        # cls_id = 1

                        xml_box = obj.find('bndbox')
                        b = (float(xml_box.find('xmin').text),
                             float(xml_box.find('xmax').text),
                             float(xml_box.find('ymin').text),
                             float(xml_box.find('ymax').text))
                        bb = self.rect2box((w, h), b)

                        # cls_id, x, y, w, h
                        obj = [int(cls_id),
                               float(bb[0]),
                               float(bb[1]),
                               float(bb[2]),
                               float(bb[3])]
                        # print(obj)
                        objs_gt.append(obj)
        return objs_gt

    def overlap(self, x1, w1, x2, w2):
        """
        :param x1:
        :param w1:
        :param x2:
        :param w2:
        :return:
        """
        l1 = x1 - w1 / 2.
        l2 = x2 - w2 / 2.
        left = l1 if l1 > l2 else l2
        r1 = x1 + w1 / 2.
        r2 = x2 + w2 / 2.
        right = r1 if r1 < r2 else r2
        return right - left

    def box_intersection(self, box_1, box_2):
        """
        :param box_1:
        :param box_2:
        :return:
        """
        w = self.overlap(box_1[0], box_1[2], box_2[0], box_2[2])
        h = self.overlap(box_1[1], box_1[3], box_2[1], box_2[3])
        if w < 0 or h < 0:
            return 0
        area = w * h
        return area

    def box_union(self, box_1, box_2):
        """
        :param box_1:
        :param box_2:
        :return:
        """
        i = self.box_intersection(box_1, box_2)
        u = box_1[2] * box_1[3] + box_2[2] * box_2[3] - i
        return u

    def box_iou(self, box_1, box_2):
        """
        :param box_1:
        :param box_2:
        :return:
        """
        return self.box_intersection(box_1, box_2) / self.box_union(box_1, box_2)

    def box2rect(self, box, width, height):
        """
        :param box:
        :param width:
        :param height:
        :return:
        """
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        left = (x - w / 2.0) * width
        top = (y - h / 2.0) * height
        right = (x + w / 2.0) * width
        bottom = (y + h / 2.0) * height
        return [int(left), int(top), int(right), int(bottom)]

    def rect2box(self, size, box):
        """
        box2rect
        :param size:
        :param box:
        :return:
        """
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = abs(box[1] - box[0])
        h = abs(box[3] - box[2])
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def comp_data(self,
                  det_dir,
                  objs_det,
                  objs_gt,
                  prob_thresh,
                  iou_thresh,
                  img_dir,
                  file_name):
        """
        :param det_dir:
        :param objs_det:
        :param objs_gt:
        :param prob_thresh:
        :param iou_thresh:
        :param img_dir:
        :param file_name:
        :return:
        """
        # print('=> do false miss? ', self.do_false_miss)

        img = cv2.imread('%s/%s.jpg' % (img_dir, file_name))
        H, W, _ = img.shape

        if self.do_false_miss:
            dets_flag = [False for n in range(0, len(objs_det))]

        show_result_path = '%s/show_result/%s_r.jpg' % (det_dir, file_name)

        correct = 0
        detect_num = 0
        total_iou = 0

        # 读取每一个标签ground truth
        for obj_gt in objs_gt:
            # cls_id, x, y, w, h
            box_gt = [obj_gt[1], obj_gt[2], obj_gt[3], obj_gt[4]]  # x, y, w, h

            if self.do_false_miss:
                rect_gt = self.box2rect(box_gt, W, H)

                # 画每一个标注框, 绿色: ground truth
                cv2.rectangle(img=img,
                              pt1=(rect_gt[0], rect_gt[1]),
                              pt2=(rect_gt[2], rect_gt[3]),
                              color=(0, 255, 0),
                              thickness=2)

            # 获取detection中与ground truth IOU最大的detection
            best_iou = 0.0
            rect_det = []
            best_idx = -1

            # 读取每一个检测结果, 取与GT的IOU最大的detection
            for idx, obj_det in enumerate(objs_det):
                # class_id, prob, x, y, w, h
                box_det = [obj_det[2], obj_det[3], obj_det[4], obj_det[5]]

                # 计算IOU
                det_iou = self.box_iou(box_gt, box_det)

                if det_iou > best_iou:
                    best_idx = idx
                    best_iou = det_iou
                    rect_det = self.box2rect(box_det, W, H)

            total_iou += best_iou

            # 如果对于此GT, 无任何有效iou检出
            if best_idx == -1:
                continue

            # 置信度、iou、检测类别的判断
            if objs_det[best_idx][1] > prob_thresh:  # 蓝色: 正确检出, True positive
                # 如果满足置信度(prob)要求
                detect_num += 1

                # iou和检测类别判断: 如果只有1类, 不用判断类别是否正确
                if best_iou > iou_thresh and \
                    int(objs_det[best_idx][0]) == obj_gt[0]:
                    correct += 1

                    if self.do_false_miss:
                        dets_flag[best_idx] = True

                        cv2.rectangle(img=img,
                                      pt1=(rect_det[0], rect_det[1]),
                                      pt2=(rect_det[2], rect_det[3]),
                                      color=(255, 0, 0),
                                      thickness=2)
                        cv2.putText(img=img,
                                    text=str(objs_det[best_idx][1]),
                                    org=(rect_det[0], rect_det[1]),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1.3,
                                    color=(255, 255, 255),
                                    thickness=2)

                else:  # 红色(画GT): 漏检, 如果不满足IOU要求: miss
                    if self.do_false_miss:
                        cv2.rectangle(img=img,
                                      pt1=(rect_gt[0], rect_gt[1]),
                                      pt2=(rect_gt[2], rect_gt[3]),
                                      color=(0, 0, 255),
                                      thickness=2)

        # --------------------------------------------------------------

        # 根据det_flag和object probability判定是否虚检
        # detect_num = 0
        for i, obj_det in enumerate(objs_det):

            if self.do_false_miss:
                box_det = [obj_det[2], obj_det[3], obj_det[4], obj_det[5]]

                if not dets_flag[i]:  # false
                    if obj_det[1] > prob_thresh:  # positive
                        rect_det = self.box2rect(box_det, W, H)

                        # 黑色: 虚检, false positive
                        cv2.rectangle(img=img,
                                      pt1=(rect_det[0], rect_det[1]),
                                      pt2=(rect_det[2], rect_det[3]),
                                      color=(0, 0, 0),
                                      thickness=2)

                        # 绘制object probability
                        txt = str(obj_det[1])
                        cv2.putText(img=img,
                                    text=txt,
                                    org=(rect_det[0], rect_det[1]),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1.3,
                                    color=(255, 255, 255),
                                    thickness=2)

                        self.false_miss.append(show_result_path)
                        # print('=> add fale miss: %s, %d'
                        #       % (show_result_path, len(self.false_miss)))

        if self.do_false_miss:
            # 每一张画过框的图存入服务器磁盘
            cv2.imwrite(filename='%s/show_result/%s_r.jpg' % (det_dir, file_name),
                        img=img)

        # ----------------------------------------------
        tp = correct
        fp = detect_num - tp
        tn = 0
        fn = len(objs_gt) - tp

        avg_iou = 0
        recall = 0
        accuracy = 0
        precision = 0

        if 0 == len(objs_gt):
            avg_iou = 0
            recall = 1
            accuracy = 1 if detect_num == 0 else 0
            precision = 1 if detect_num == 0 else 0
        else:
            avg_iou = total_iou / len(objs_gt)
            recall = correct / float(len(objs_gt))

            try:
                accuracy = correct / float(tp + fn + fp + tn)
            except Exception as e:
                print('=> tp: ', tp, 'fn: ', fn, 'fp:', fp, 'tn: ', tn)

            # 检测正确数大于检测结果数的情况，即同一个目标多次标记
            corr = (correct if correct < detect_num else detect_num)

            precision = 0.0 if detect_num == 0 else corr / float(detect_num)

        cmp_res = [file_name,
                   len(objs_gt),
                   detect_num,
                   correct,
                   recall,
                   avg_iou,
                   accuracy,
                   precision]

        # whether to do false miss
        if self.do_false_miss:
            # print(len(self.false_miss))
            return cmp_res, self.false_miss
        else:
            return cmp_res, None

    def export2excel(self,
                     comp_result,
                     total_result,
                     result_path):
        """
        :param comp_result:
        :param total_result:
        :param result_path:
        :return:
        """
        # 创建工作簿
        f = xlwt.Workbook()
        sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

        row0 = [u'图片名', u'标注目标', u'检测目标', u'检测正确',
                u'recall', u'iou', u'accuracy', u'precision']
        for i in range(0, len(row0)):
            sheet1.write(0, i, row0[i])

        for r in range(0, len(comp_result)):
            for c in range(0, len(comp_result[r])):
                sheet1.write(r + 1, c, comp_result[r][c])

        row_end = [u'total',
                   total_result[0],
                   total_result[1],
                   total_result[2],
                   total_result[3],
                   total_result[4],
                   total_result[5],
                   total_result[6]]
        for i in range(0, len(row_end)):
            sheet1.write(len(comp_result) + 2, i, row_end[i])

        save_path = '%s/AnalyseResult.xls' % (result_path)
        f.save(save_path)
        print('=> analysis result %s saved.' % save_path)

    def run_statistics(self,
                       prob_thresh=0.4,
                       iou_thresh=0.5):
        """
        :param prob_thresh:
        :param iou_thresh:
        :return:
        """
        # 读取标注数据和检测结果并统计PR等参数
        total_label = 0
        total_detect = 0
        total_corr = 0
        total_iou = 0
        file_num = 0
        cmp_result = []

        for k, f_name in enumerate(self.f_list):
            # 记录正在处理的百分比
            self.cursor = int(float(k) / float(len(self.f_list)) * 100.0 + 0.5)

            # 发射sample cursor改变信号
            self.cursor_signal.signal.emit()

            # print('=> cursor: ', self.cursor)
            file_num += 1

            # 检查文件是否存在并打开
            result_file = '%s/%s.txt' % (self.det_dir, f_name)
            label_file = '%s/%s.xml' % (self.anno_dir, f_name)
            if not (os.path.exists(result_file) and os.path.exists(label_file)):
                print("[Warning]: label or result file is not exist")
                continue

            # class_id, prob, x, y, w, h
            f_result = open(result_file, 'r')
            f_label = open(label_file)

            # 加载检测结果
            objs_det = self.load_result(self.det_dir, f_result)

            # 加载标注数据
            objs_gt = self.load_label(self.anno_dir,
                                      f_label,
                                      f_name,
                                      self.target_type)
            total_label += len(objs_gt)

            # 释放文件资源
            f_label.close()
            f_result.close()

            # 比较每张图片的检测结果和label标记数据, 默认0.45, 0.6
            cmp, false_miss = self.comp_data(self.det_dir,
                                             objs_det,
                                             objs_gt,
                                             prob_thresh,
                                             iou_thresh,
                                             self.img_dir,
                                             f_name)

            total_corr += cmp[3]
            total_iou += cmp[5] * cmp[1]

            cmp_result.append(cmp)
            if k % 50 == 0:
                print('%5d  label: %2d  detect: %2d  correct: %2d'
                      '  recall: %.2f  avg_iou: %.2f  accr: %.2f  prec: %.2f'
                      % (file_num, cmp[1], cmp[2], cmp[3],
                         cmp[4], cmp[5], cmp[6], cmp[7]))
            total_detect += cmp[2]

        # 统计分析结果
        total_result = [total_label,
                        total_detect,
                        total_corr,
                        total_corr / float(total_label),
                        total_iou / total_label,
                        float(total_corr) / (total_label + total_detect - total_corr),
                        float(total_corr) / total_detect]

        # output final statistics result
        # self.export2excel(cmp_result, total_result, self.det_dir)

        print('\n=> total_label: %d  total_detect: %d  total_corr: %d'
              '  recall: %.2f  avg iou: %.2f  accuracy: %.2f  precision: %.2f ' % \
              (total_result[0], total_result[1], total_result[2],
               total_result[3], total_result[4], total_result[5], total_result[6]))
        print('=> total %d analyse_testset image files.' % file_num)

        return total_result[6], total_result[3], false_miss

    def run_PR(self):
        """
        :return:
        """
        if not os.path.isdir(self.test_root):
            print('=> invalid test-set root.')
            return

        print('=> processing test set: ', self.test_root)

        # 求PRs
        self.PRs = []  # 清空PR
        the_P = []
        the_R = []

        self.cycle = 0
        max_PxR = 0.0
        the_thresh = 0.0

        for prob_th in range(25, 86):
            self.pr_idx = int(float(prob_th - 25) / 60.0 * 100.0 + 0.5)
            # print('=> pr_idx: ', self.pr_idx)

            # 发射pr_idx改变信号
            self.pr_idx_signal.signal.emit()

            self.prob_th = prob_th * 0.01

            P, R, _ = self.run_statistics(prob_thresh=self.prob_th,
                                          iou_thresh=0.5)
            this_PxR = float(P) * float(R)
            if this_PxR > max_PxR:
                max_PxR = this_PxR
                the_P, the_R = [P], [R]
                the_thresh = prob_th * 0.01

            # 更新当前PR值
            self.PR = [P, R]
            # print('=> self.PR: ', self.PR)

            # 发射一次PR计算结束的信号
            self.ana_done_signal.signal.emit()

            print('=> prob_th: {:.3f}, p_r: {:.3f}%, {:.3f}%\n'
                  .format(self.prob_th, P * 100.0, R * 100.0))
            self.PRs.append((P, R))
            self.cycle += 1

        # TODO: 将PRs列表序列化并输出
        print('=> len(self.PRs): %d' % len(self.PRs))
        print('self.PRs[:5]:\n', self.PRs[:5])

        PRs_path = self.test_root + '/PRs.pkl'
        with open(PRs_path, 'wb') as f_h:
            pickle.dump(self.PRs, f_h)
            print('=> PRs dumped @%s' % PRs_path)

        X = [r for p, r in self.PRs]
        Y = [p for p, r in self.PRs]
        ylim(min(Y) * 0.99, 0.99)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(X, Y)

        # TODO: 将P*R值最大的点醒目显示(包括Threshold和P, R)
        plt.scatter(the_R, the_P, s=120, c='g')

        # TODO: 将P, R值的文本绘制到PR曲线图上
        label = '%.3f' % the_P[0] \
                + ' %.3f' % the_R[0] \
                + ', %.2f' % the_thresh
        print('=> max P*R info: ', label)
        plt.text(the_R[0], the_P[0], label)

        # 保存PR曲线图
        self.pr_img_path = self.test_root + '/' + 'car_pr.png'
        plt.savefig(self.pr_img_path)
        print('=> %s saved' % self.pr_img_path)
        # plt.show()

    def analyse_testset(self,
                        prob_thresh=0.45):
        """
        :param prob_thresh:
        :return:
        """
        if not os.path.isdir(self.test_root):
            print('=> invalid test-set root.')
            return

        # 统计
        self.prob_th = prob_thresh
        P, R, false_miss = self.run_statistics(prob_thresh=self.prob_th,
                                               iou_thresh=0.5)

        if self.do_false_miss:
            # clear old data
            false_miss_dir = self.output_dir + '/' + 'false_miss'
            if not os.path.exists(false_miss_dir):
                os.makedirs(false_miss_dir)
            for x in os.listdir(false_miss_dir):
                x_path = os.path.join(false_miss_dir, x)
                if x_path.endswith('.jpg'):
                    os.remove(x_path)

            if false_miss is not None:
                # 移动拷贝数据false and miss数据
                print('=> copy and move false_miss images...')
                for i, item in enumerate(false_miss):
                    dst_path = os.path.join(false_miss_dir, os.path.split(item)[1])

                    print(dst_path)
                    if not os.path.exists(dst_path):
                        shutil.move(item, false_miss_dir)
                        print('=> %s moved to %s' % (false_miss[i], false_miss_dir))
                print('=> false miss images done.')

        return P, R


# ----------------------------------------------------

class MainWindow(QMainWindow):  # QMainWindow
    """
    主窗口
    """

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_dirs()

    def init_dirs(self):
        """
        define target type
        :return:
        """
        # ['bkg', 'car'] or ['bkg', 'car_front', 'car_rear']
        self.target_type = ['bkg', 'car']  # car
        # self.target_type = ['fr', 'car', 'bicycle',
        #                     'person', 'cyclist', 'tricycle']
        self.analyser = Analyser(test_root='',
                                 target_type=self.target_type)

        # 为Analyser注册进度条更新槽函数
        self.analyser.cursor_signal.signal.connect(self.update_prog_bar_1)
        self.analyser.pr_idx_signal.signal.connect(self.update_prog_bar_2)

        # 为Analyser注册分析结果槽函数
        self.analyser.ana_done_signal.signal.connect(self.show_pr)

    def init_ui(self):
        self.setGeometry(300, 300, 585, 650)
        self.setWindowTitle('TestsetAnalyser')

        # 输入提示
        self.lb_2 = QLabel('analysis result...', self)
        self.lb_2.move(270, 80)
        self.lb_2.adjustSize()

        # PR曲线图显示
        self.lb_3 = QLabel(self)
        self.lb_3.setGeometry(25, 130, 550, 550)
        self.lb_4 = QLabel('test root: ', self)
        self.lb_4.move(150, 30)

        # 按钮
        self.btn_1 = QPushButton('Run analysis', self)
        self.btn_1.move(20, 80)
        self.btn_2 = QPushButton('Run PR', self)
        self.btn_2.move(20, 130)
        self.btn_3 = QPushButton('Pick dir', self)
        self.btn_3.move(20, 30)
        self.btn_4 = QPushButton('False Miss', self)
        self.btn_4.move(385, 130)

        # 进度条
        self.prog_bar_1 = QProgressBar(self)
        self.prog_bar_1.move(150, 80)
        self.prog_bar_1.setOrientation(Qt.Horizontal)

        self.prog_bar_2 = QProgressBar(self)
        self.prog_bar_2.move(150, 130)
        self.prog_bar_2.setOrientation(Qt.Horizontal)

        # 单选框
        self.rb = QRadioButton('Is false miss', self)
        self.rb.move(270, 130)

        # 单选框槽函数
        self.rb.toggled.connect(self.rb_checked)

        # 按钮槽函数
        self.btn_1.clicked.connect(self.btn_1_pressed)
        self.btn_2.clicked.connect(self.btn_2_pressed)
        self.btn_3.clicked.connect(self.pick_dir)
        self.btn_4.clicked.connect(self.open_fm_dir)

        self.show()

    def open_fm_dir(self):
        """
        打开false miss目录
        :return:
        """
        if not self.analyser.do_false_miss:
            print('=> false miss not done.')
            return

        if os.path.isdir(self.analyser.test_root):
            fale_miss_dir = self.analyser.output_dir + '/' + 'false_miss'
            os.system('nautilus ' + fale_miss_dir)
            # os.system('sudo nautilus ' + fale_miss_dir)

    def pick_dir(self):
        """
        选择指定
        :return:
        """
        dir = QFileDialog.getExistingDirectory(self,
                                               'Pick dir',
                                               '/users')
        if os.path.isdir(dir):
            print('=> choosen dir: ', dir)
            self.analyser.set_test_root(dir)
            self.lb_4.setText('test root: ' + self.analyser.test_root)
            self.lb_4.adjustSize()

    def update_prog_bar_1(self):
        """
        进度条更新事件响应
        :return:
        """
        self.prog_bar_1.setValue(self.analyser.cursor)

    def update_prog_bar_2(self):
        """
        :return:
        """
        self.prog_bar_2.setValue(self.analyser.pr_idx)

    def show_pr(self):
        """
        显示分析结果
        :return:
        """
        # print(self.analyser.PR)
        self.lb_2.setText('cycle %d | ' % (self.analyser.cycle + 1)
                          + 'prob_th %.2f | ' % (self.analyser.prob_th)
                          + ' PR: ' + '{:.2%}'.format(self.analyser.PR[0])
                          + ', ' + '{:.2%}'.format(self.analyser.PR[1]))
        self.lb_2.adjustSize()

    def rb_checked(self):
        """
        是否统计False miss数据并拷贝到独立子目录
        :return:
        """
        if self.rb.isChecked():
            self.analyser.do_false_miss = True
            print('=> do false miss checked.')

    def btn_2_pressed(self,
                      do_false_miss):
        """
        :param do_false_miss:
        :return:
        """
        # 设置test root
        if os.path.isdir(self.analyser.test_root):
            # PR曲线
            self.analyser.run_PR()

            if os.path.isfile(self.analyser.pr_img_path) \
                    and self.analyser.pr_img_path != '':
                img = QPixmap(self.analyser.pr_img_path)
                img = img.scaled(530, 450)
                self.lb_3.setPixmap(img)
        else:
            print('=> invalid test root')
            # TODO:显示WARNING对话框

    def btn_1_pressed(self):
        """
        分析一组数据
        :return:
        """
        # 业务代码...
        if os.path.isdir(self.analyser.test_root):
            print('=> processing test root: ', self.analyser.test_root)

            self.analyser.PR = self.analyser.analyse_testset()

            # 发射一次PR计算结束的信号
            self.analyser.ana_done_signal.signal.emit()
        else:
            # TODO:显示WARNING对话框
            print('=> invalid test root!')
            return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    sys.exit(app.exec_())

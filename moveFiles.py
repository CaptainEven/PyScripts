# _*_coding: utf-8_*_

'''
一定要写一个脚本解决文件拷贝重命名的问题
'''
import os

def move(src_path, dst_path, ext, count):
    if not os.path.exists(src_path):
        print('[Error]: src dir not exists.')
        return
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        f_list = os.listdir(src_path)
        my_count = 0
        for f in f_list:
            f_name = os.path.splitext(f)
            if f_name[1] == ext:
                pass


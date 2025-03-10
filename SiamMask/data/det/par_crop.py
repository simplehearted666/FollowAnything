# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join, isdir
from os import mkdir, makedirs
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import time
import sys


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_like_SiamFCx(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return x


def crop_xml(xml, sub_set_crop_path, instanc_size=511):
    # 解析 XML 文件
    xmltree = ET.parse(xml)
    # 查找所有的 object 标签
    objects = xmltree.findall('object')

    # 构建当前 XML 文件对应的裁剪图像保存路径
    frame_crop_base_path = join(sub_set_crop_path, xml.split('/')[-1].split('.')[0])
    # 如果路径不存在，则创建该路径
    if not isdir(frame_crop_base_path):
        makedirs(frame_crop_base_path)

    # 根据 XML 文件路径推断出对应的图像路径
    img_path = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')
    # 读取图像
    im = cv2.imread(img_path)
    # 计算图像各通道的平均像素值
    avg_chans = np.mean(im, axis=(0, 1))

    for id, object_iter in enumerate(objects):
        # 查找边界框信息
        bndbox = object_iter.find('bndbox')
        # 提取边界框的坐标信息
        bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]

        # 调用裁剪函数进行裁剪操作
        x = crop_like_SiamFCx(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        # 将裁剪后的图像转换为 CV_8U 格式，避免 cv::imwrite 警告
        x_8u = cv2.convertScaleAbs(x)
        # 保存裁剪后的图像到指定路径
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, id)), x_8u)


def main(instanc_size=511, num_threads=24):
    crop_path = './crop{:d}'.format(instanc_size)
    if not isdir(crop_path): mkdir(crop_path)
    VID_base_path = './ILSVRC2015'
    ann_base_path = join(VID_base_path, 'Annotations/DET/train/')
    sub_sets = ('ILSVRC2013_train', 'ILSVRC2013_train_extra0', 'ILSVRC2013_train_extra1', 'ILSVRC2013_train_extra2', 'ILSVRC2013_train_extra3', 'ILSVRC2013_train_extra4', 'ILSVRC2013_train_extra5', 'ILSVRC2013_train_extra6', 'ILSVRC2013_train_extra7', 'ILSVRC2013_train_extra8', 'ILSVRC2013_train_extra9', 'ILSVRC2013_train_extra10', 'ILSVRC2014_train_0000', 'ILSVRC2014_train_0001','ILSVRC2014_train_0002','ILSVRC2014_train_0003','ILSVRC2014_train_0004','ILSVRC2014_train_0005','ILSVRC2014_train_0006')
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        if 'ILSVRC2013_train' == sub_set:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
        else:
            xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))

        n_imgs = len(xmls)
        sub_set_crop_path = join(crop_path, sub_set)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_xml, xml, sub_set_crop_path, instanc_size) for xml in xmls]
            for i, f in enumerate(futures.as_completed(fs)):
                printProgress(i, n_imgs, prefix=sub_set, suffix='Done ', barLength=80)


if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

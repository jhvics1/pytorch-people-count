# -*- coding: utf-8 -*-
from __future__ import division

import argparse
import shutil

from utils.utils import *
from utils.datasets import *
from utils.load_data_V2 import *
from network.object_detection.models import *
from network.crowd_couting.SDCNet import SDCNet_VGG16_classify
import mysql.connector


def db_insert(opt, item_dict):
    sql = "INSERT INTO {table} (filename, htag, model, result) VALUES (%s, %s, %s, %s)".format(table=opt.db_table)
    item_dict['htag'] = "haha"
    # ex) result = "sdcnet|123"
    result = item_dict['network'] + "|" + str(item_dict['count'])
    value = (item_dict['filename'],
             item_dict['htag'],
             0,
             result)
    opt.mycursor.execute(sql, value)


class Yolo:
    def __init__(self, opts):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.opts.model_def = "config/yolov3.cfg"
        self.opts.weights_path = "weights/yolo/yolov3.weights"
        self.opts.class_path = "data/coco.names"
        self.opts.conf_thres = 0.8
        self.opts.nms_thres = 0.4
        self.opts.img_size = 416

        self.model = Darknet(self.opts.model_def, img_size=self.opts.img_size).to(device)

        if self.opts.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.opts.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.opts.weights_path))
        # Set in evaluation mode
        self.model.eval()

        self.classes = load_classes(self.opts.class_path)  # Extracts class labels from file
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def get_count(self, path):
        result = dict()
        head_cnt = 0
        result['count'] = 0
        result['network'] = "yolo"
        input_imgs = transform_img_yolo(path, self.opts.img_size)
        input_imgs = Variable(input_imgs.type(self.Tensor))
        input_imgs = torch.unsqueeze(input_imgs, 0)
        # Get detections
        with torch.no_grad():
            # print("shape - {}".format(input_imgs.shape))
            if input_imgs.shape[1] > 1:
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, self.opts.conf_thres, self.opts.nms_thres)
            else:
                print("shape - {}, Path {}".format(input_imgs.shape, path))

            for i, item in enumerate(detections):
                if item is not None:
                    # print("detections 2 - {}".format(detections))
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in item:

                        if self.classes[int(cls_pred)] == 'person':
                            head_cnt += 1
                            result['count'] += 1
                    # filename = img_paths[0].split("/")[-1].split(".")[0]
                    filename = path.split("/")[-1]
                    result['filename'] = filename
                    log_str = '{\'filename\':\'%s\', \'count\':%.3f}' % (result['filename'], result['count'])
                    print("\t--> Log (Yolo) - \t{}".format(log_str))
                    if self.opts.log_enable:
                        txt_write(self.opts.output + "/output.log", log_str, mode='a')
                    if self.opts.db_enable:
                        db_insert(self.opts, result)
        return ('%.3f'%result['count'])


class SDCNet:
    def __init__(self, args):
        self.args = args
        self.Tensor = ToTensor_2
        self.opts = dict()
        self.opts['partition'] = 'two_linear'
        self.opts['step'] = 0.5
        self.opts['max_num'] = 7
        self.opts['psize'], self.opts['pstride'] = 64, 64

        # set label_indice
        if self.opts['partition'] == 'one_linear':
            label_indice = np.arange(self.opts['step'], self.opts['max_num'] + self.opts['step'] / 2, self.opts['step'])
            add = np.array([1e-6])
            label_indice = np.concatenate((add, label_indice))
        elif self.opts['partition'] == 'two_linear':
            label_indice = np.arange(self.opts['step'], self.opts['max_num'] + self.opts['step'] / 2, self.opts['step'])
            add = np.array([1e-6, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
            label_indice = np.concatenate((add, label_indice))
        self.mod_path = 'weights/sdcnet/SHB/best_epoch.pth'
        self.opts['label_indice'] = label_indice
        self.opts['class_num'] = label_indice.size + 1
        self.label_indice = torch.Tensor(label_indice)
        self.class_num = len(label_indice) + 1
        self.model = SDCNet_VGG16_classify(self.class_num, self.label_indice, psize=self.opts['psize'],
                                           pstride=self.opts['pstride'], div_times=2, load_weights=True).cuda()

        all_state_dict = torch.load(self.mod_path)
        self.model.load_state_dict(all_state_dict['net_state_dict'])
        self.rgb_dir = "weights/sdcnet/SHB/rgbstate.mat"
        self.rgb_dir = os.path.abspath(self.rgb_dir)
        self.mat = sio.loadmat(self.rgb_dir)
        self.rgb = self.mat['rgbMean'].reshape(1, 1, 3)

    def resize_with_ratio(self, image):
        target_w = 1366 
        target_h = 768
        # print('image shape - {}'.format(image.shape))
        h = image.shape[0]
        w = image.shape[1]
        if h > w:
            resize_h = target_h
            resize_w = int(h * target_w / target_h)
        else:
            resize_w = target_w
            resize_h = int(w * target_h / target_w)
        
        return cv2.resize(image, (resize_w, resize_h))

    def get_count(self, path):
        result = dict()
        result['network'] = "sdcnet"
        image = io.imread(path)
        image = self.resize_with_ratio(image)
        if len(image.shape) > 2:
            image = image / 255. - self.rgb  # to normalization,auto to change dtype
        else:
            # print("shape - {}, Path {}".format(image.shape, path))
            image = cv2.cvtColor(path, cv2.COLOR_GRAY2RGB)
            image = image / 255. - self.rgb
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        image = get_pad(image, DIV=64)
        image = image.type(torch.float32)
        # print('image shape - {}'.format(image.shape))
        image = image.cuda()
        input_imgs = torch.unsqueeze(image, 0)

        # print("Input shape - {}".format(input_imgs.shape))
        features = self.model(input_imgs)
        div_res = self.model.resample(features)
        merge_res = self.model.parse_merge(div_res)
        outputs = merge_res['div' + str(self.model.args['div_times'])]
        del merge_res

        result['count'] = (outputs).sum().item()
        filename = path.split("/")[-1]
        result['filename'] = filename
        log_str = '{\'filename\':\'%s\', \'count\':%.3f}' % (result['filename'], result['count'])
        print("\t--> Log (SDCNet) - \t{}".format(log_str))
        if self.args.log_enable:
            txt_write(self.args.output + "/output.log", log_str, mode='a')
        if self.args.db_enable:
            db_insert(self.args, result)
        return ('%.3f'%result['count'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--log_enable", action='store_true', default=False, help='Store result in a log file')
    parser.add_argument("--output", type=str, default='output', help="path to checkpoint model")
    parser.add_argument("--db_enable", action='store_true', default=False, help='Insert prediction result to db')
    parser.add_argument("--db_host", type=str, default='localhost', help="URI of db, ex) 192.168.0.1")
    parser.add_argument("--db_user_name", type=str, default='USER_NAME', help="db user name")
    parser.add_argument("--db_user_pw", type=str, default='USER_PW', help="db user password")
    parser.add_argument("--db_name", type=str, default='DB_NAME', help="Name of db")
    parser.add_argument("--db_table", type=str, default='TABLE_NAME', help="Name of db table to insert")

    args = parser.parse_args()
    print(args)

    model = dict()
    model['yolo'] = Yolo(args)
    model['sdcnet'] = SDCNet(args)

    shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    if args.log_enable:
        txt_write(self.args.output + "/output.log", log_str, mode='w')

    if args.db_enable:
        args.mydb = mysql.connector.connect(
            host=args.db_host,
            user=args.db_user_name,
            passwd=args.db_user_pw,
            database=args.db_name
        )
        args.mycursor = args.mydb.cursor()

    for img in sorted(os.listdir(args.image_folder)):
        # print("img - {}".format(img))
        img = os.path.join(args.image_folder, img)
        # Yolo version
        val = model['yolo'].get_count(img)
        print('{}'.format(val))
        # SDCNet Version
        val = model['sdcnet'].get_count(img)
        print('{}'.format(val))

    if args.db_enable:
        args.mydb.commit()

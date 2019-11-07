from config import parse_args
from utils import mkdir, load_image_names_and_path, Display_remain_time
from face_preprocess import pre_process
import cv2
import time
import numpy as np

def show_details(args, n_total_img):
    print('\n------------------ Summary ---------------------')
    if args.face_de_platform == 'MXNET':
        print(' Face Detector: You are using RetinaFace-Res50 (MXNET) which is more accurate but slower')
    elif args.face_de_platform == 'Pytorch':
        print(' Face Detector: You are using RetinaFace-BobileNet (Pyotrch) which is less accurate but faster')
    print(' The input img folder is: ' + args.input_folder)
    print(' The aligned img will save in: ' + args.output_folder)
    print(' # of image is: ' + str(n_total_img))
    if args.output_format == '218*178':
        print(' The output image format will be 218*178, save format as CelebA and Megaage')
    elif args.output_format == '178*178':
        print(' The output image format will be 178*178')
    print('------------------ Summary ---------------------\n')

def reform_detector(bbox_cs, points):
    x1, y1, x2, y2 = bbox_cs[0, 0:4]
    w = x2 - x1
    h = y2 - y1
    box = [x1, y1, w, h]
    points_new = np.zeros((5,2))
    points_new = points[0,:,:]
    return box, points_new

def inference(args):
    # build detector
    if args.face_de_platform == 'MXNET':
        # Retina-Face Detector - MXNET - Res50
        from RetinaFace_Detector.retinaface import RetinaFace
        detector = RetinaFace(args.weight_detect_retina, 0, args.gpuid_Retina, 'net3')
    elif args.face_de_platform == 'Pytorch':
        # Retina-Face Detector - Pytorch - MBv2
        from RetinaFace_pytorch.detector import Retinaface_Detector
        detector = Retinaface_Detector(args, use_gpu=True)

    # get img paths from the folder
    image_names, image_paths, image_names_no_suffix = load_image_names_and_path(args.input_folder)
    n_total_img = len(image_paths)

    # output folder
    mkdir(args.output_folder)

    # show detail of running
    show_details(args, n_total_img)

    # remain time calculate
    time_start = time.time()
    num_done_for_display = 0

    n_saved_img = 0
    # loop
    for i in range(n_total_img):
        image_path = image_paths[i]
        image_name = image_names[i]
        input_img = cv2.imread(image_path)
        save_path = args.output_folder + '/' + image_name

        # get bbox and points
        if args.face_de_platform == 'MXNET':
            bbox_cs, points = detector.detect(input_img, args.fa_de_thresh, scales=args.scales, do_flip=args.do_flip)
        elif args.face_de_platform == 'Pytorch':
            [bbox_cs, points] = detector.detect(input_img)

        if bbox_cs is not None and bbox_cs.shape[0] != 0:
            bbox, points = reform_detector(bbox_cs, points)

            # alignment
            img_218_178, img_178_178 = pre_process(input_img, bbox, points)

            # save
            # u can modify this part to save your own format of img after the alignment.
            if args.output_format == '218*178':
                cv2.imwrite(save_path, img_218_178)
            elif args.output_format == '178*178':
                cv2.imwrite(save_path, img_178_178)

            # count saved number of images
            n_saved_img += 1

        # display remain time
        num_done_for_display = Display_remain_time(num_done_for_display, n_total_img, time_start)
    print('\nDone! Total sucessfully detected and aligned img is: {}/{} ({:2.2f}%).'.format(n_saved_img, n_total_img, n_saved_img/n_total_img*100))

if __name__ == '__main__':
    args = parse_args()
    inference(args)
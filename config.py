import argparse

def parse_args():
    parser = argparse.ArgumentParser(
     description='For face dataset preprocess, including face detection and alignment - Tianchi.L')

    # choose face detector
    parser.add_argument('--face_de_platform', default='MXNET', type=str, choices=['MXNET', 'Pytorch'])
    # input and output path
    parser.add_argument('--input_folder', required=True, type=str, help='The path of the folder of input images')
    parser.add_argument('--output_folder', required=True, type=str, help='The path of the folder of output images')
    parser.add_argument('--output_format', required=True, choices=['218*178', '178*178'], help='The path of the folder of output images')

    # face detector parameters
    parser.add_argument('--weight_detect_retina', default='./RetinaFace_Detector/model/R50', help='face_detector_model_path')
    parser.add_argument('--gpuid_Retina', default=0, type=int, help='gpuid_Retina')
    parser.add_argument('--fa_de_thresh', default=0.30, type=float, help='threshold of face detector')
    parser.add_argument('--scales', default=[1.0])
    parser.add_argument('--do_flip', default=False)

    parser.add_argument('--fa_de_target_size', default=360, type=int, choices=[360, 720, 1080])


    return parser.parse_args()

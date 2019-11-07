import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable
#from retinaface import load_retinaface_mbnet
from RetinaFace_pytorch.utils import RetinaFace_Utils
from RetinaFace_pytorch.retinaface import RetinaFace_MobileNet
def load_retinaface_mbnet():
    net = torch.load('Face_Attributes/RetinaFace_pytorch/retina_pb.pt')
    #net.load_state_dict(checkpoint)
    net.eval()
    return net

class Retinaface_Detector(object):
    def __init__(self, args, use_gpu=True):
        self.args = args
        self.threshold = self.args.fa_de_thresh
        self.use_gpu = use_gpu
        self.model = RetinaFace_MobileNet()
        self.model.load_state_dict(torch.load('.//RetinaFace_pytorch/retina_pb.pth', map_location='cuda:0'))
        # def adapatation(m):
        #     if isinstance(m, torch.nn.Linear):
        #         if m.bias is None:
        #             print('None bias 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #             m.bias = torch.nn.Parameter(torch.zeros(m.out_feature).cuda())
        # self.model.apply(adapatation)
        self.model.cuda()
        self.model.eval()



        # ('Face_Attributes/RetinaFace_pytorch/retina_pb.pt')
        # self.model = load_retinaface_mbnet().cuda() if use_gpu else load_retinaface_mbnet()
        self.pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.pixel_scale = float(1.0)
        self.utils = RetinaFace_Utils()
        # torch.save(self.model,'./retina_pb.tb')

    def convert_2_tensorRT(self, module_name, example_data):
        if module_name == 'retinaface_res18':
            from torch.autograd import Variable
            from torch2trt import torch2trt
            # from PIL import Image
            # from torchvision import transforms
            # img_218_178 = torch.ones((1, 3, 218, 178)).float()
            # transform = []
            # transform.append(transforms.Resize(size=(224, 224)))  # test no resize operation.
            # transform.append(transforms.ToTensor())
            # transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            # transform = transforms.Compose(transform)
            # PIL_image = Image.fromarray(img_218_178)
            # example_data = transform(PIL_image)
            # example_data = Variable(example_data).type(torch.FloatTensor).cuda()
            model_trt = torch2trt(self.model, [example_data], fp16_mode=True, strict_type_constraints=True)
            y = self.model(example_data)
            y_trt = model_trt(example_data)
            print('diff', torch.max(torch.abs(y - y_trt)))
            print('y', y)
            print('y_trt', y_trt)
            # Save
            torch.save(model_trt.state_dict(), './RetinaFace_trt_fp16_tianchi.pth')

    def img_process(self, img):
        target_size = self.args.fa_de_target_size
        max_size = 1920
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        im = im.astype(np.float32)
        # print('iii', im.shape)
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / \
                                    self.pixel_stds[2 - i]
        return im_tensor, im_scale

    def detect(self, img):
        results = []
        im, im_scale = self.img_process(img)
        im = torch.from_numpy(im)
        im_tensor = Variable(im).type(torch.FloatTensor).cuda() if self.use_gpu else Variable(im)

        # self.convert_2_tensorRT('retinaface_res18', example_data=im_tensor)

        output = self.model(im_tensor)

        faces, landmarks = self.utils.detect(im, output, self.threshold, im_scale)
        
        if faces is None or landmarks is None:
            return results
        
        for face, landmark in zip(faces, landmarks):
            face = face.astype(np.int)
            landmark = landmark.astype(np.int)
            results.append([face, landmark])
        
        return faces, landmarks
        


    def detect2(self, img):
        results = []
        im, im_scale = self.img_process(img)
        im = torch.from_numpy(im)
        im_tensor = Variable(im).cuda() if self.use_gpu else Variable(im)
        output = self.model(im_tensor)
        faces, landmarks = self.utils.detect(im, output, self.threshold, im_scale)
        
        if faces is None or landmarks is None:
            return results
        
        # for face, landmark in zip(faces, landmarks):
        #     face = face.astype(np.int)
        #     landmark = landmark.astype(np.int)
        #     results.append([face, landmark])
        
        return faces,landmarks

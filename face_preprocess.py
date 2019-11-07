
import cv2
import numpy as np
from skimage import transform as trans

def alignment_218_178(img, bbox=None, landmark=None, **kwargs):
  M = None
  image_size = []
  str_image_size = kwargs.get('image_size', '')
  if len(str_image_size)>0:
    image_size = [int(x) for x in str_image_size.split(',')]
    if len(image_size)==1:
      image_size = [image_size[0], image_size[0]]
    assert len(image_size)==2
  assert len(image_size)==2
  src = np.array([
    [70.808, 112.620],  # 'left_eye'
    [108.847, 112.430],  # 'right_eye
    [89.594, 133.812],  # 'nose'
    [73.842, 153.59],  # 'mouth_left'
    [105.3475, 153.7383] ], dtype=np.float32 )  # 'mouth_right'
  dst = landmark.astype(np.float32)
  tform = trans.SimilarityTransform()
  tform.estimate(dst, src)
  M = tform.params[0:2,:]
  assert len(image_size)==2
  warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
  # x1, y1, w, h = bbox
  # x2 = x1 + w
  # y2 = y1 + h
  # img_face = np.zeros((img.shape[0], img.shape[1], 3))
  # img_face[y1:y2+1, x1:x2+1, :] = img[y1:y2+1, x1:x2+1, :]
  # warped_b = cv2.warpAffine(img_face, M, (image_size[1], image_size[0]), borderValue=0.0)
  return warped

def pre_process(input_img, bbox, keypoint):
  # Alignment. Noted img_218_178 is following the Megaage_Asian and CelebA's format.
  img_218_178 = alignment_218_178(input_img, bbox, keypoint, image_size='218,178')  # BGR

  # resize and cut
  img_178_178 = img_218_178[20:-20, :, :]
  # img_178_178_b = img_218_178_b[20:-20, :, :]
  # img_64_64 = cv2.resize(img_178_178, (64, 64))
  # img_64_64_b = cv2.resize(img_178_178_b, (64, 64))

  return img_218_178, img_178_178

# def getfaces(TK_results, detected, track_ID_results, input_img):
#   num_of_face = min(len(TK_results), len(detected))
#   faces64 = np.empty((len(detected), 64, 64, 3))
#   faces64_b = np.empty((len(detected), 64, 64, 3))
#   for i in range(num_of_face):
#     d = TK_results[i]
#     track_ID_results[i] = int(d['track_ID'])
#
#     # get bbox and landmarks(keypoints)
#     keypoint = d['keypoints']
#     bbox = d['box']
#
#     # preprocess, including alignment, resize and cut.
#     img_64_64, img_64_64_b, img_178_178 = pre_process(input_img, bbox, keypoint)
#
#     # pack into batch
#     faces64[i, :, :, :] = img_64_64
#     faces64_b[i, :, :, :] = img_64_64_b
#
#   return faces64, faces64_b, track_ID_results



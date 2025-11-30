
from osgeo import gdal
from scipy import optimize
import os
from glob import glob
import torch
from torch.autograd import Variable
from data.option import parser
from model.HPSR import HPSRnet
from data.dataset import HPSRdataset
import numpy as np
import math
import cv2
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def psnr1(img1, img2):
   return 10. * math.log10(1. / ((img1 - img2) ** 2).mean())

model_path=r'/HPSR-x5-best.pth.tar'
imgpath=r'/dataset/val/data/'
targetpath=r'/dataset/val/target/'
savepath=r'/testdata'
imgnamelist=natsorted(glob('%s/*.npy'%imgpath0))
if not os.path.exists(savepath):
   os.makedirs(savepath)
args = parser.parse_args()
srmodel = HPSRnet(args)
checkpoint = torch.load(model_path)
state_dict = checkpoint['state_dict']#
model_dict=srmodel.state_dict()
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
model_dict.update(state_dict)
srmodel.load_state_dict(state_dict)
if torch.cuda.is_available():
    srmodel = srmodel.cuda()
for index in range(len(imgnamelist))[0:10]:
    imgpath=imgnamelist[index]
    targetpath=os.path.join(targetpath0,os.path.split(imgnamelist[index])[1])
    img0=np.load(imgpath)
    print(img0.max(),img0.min())
    img=np.load(imgpath).astype('float32')[np.newaxis,np.newaxis,:,:] 
    data=torch.from_numpy(img)
    target=np.load(targetpath)
    c=cv2.resize(img0,(0,0),fx=5,fy=5,interpolation=cv2.INTER_NEAREST)
    if torch.cuda.is_available():
        data = data.cuda()
    output0 = srmodel(data,data)
    print(output0.max(),output0.min())
    if torch.cuda.is_available():
        output=output0.cpu().detach().numpy()[0][0]
    else:
        output=output0.detach().numpy()[0][0]
    np.save(savepath+'/'+os.path.split(imgnamelist[index])[1],output)
 

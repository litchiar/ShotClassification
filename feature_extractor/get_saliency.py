import torch
from torchvision.transforms import ToTensor, ToPILImage
from feature_extractor.saliency import r3net
from PIL import Image
import os
saliency_model=None

def get_saliency_(img_list,output_list,device='cuda:0',mutliply=False,invert=False):
    global  saliency_model
    assert len(img_list)==len(output_list)
    if saliency_model is None:
        saliency_model=r3net.R3Net().to(device)
        saliency_model.load_state_dict(torch.load(r3net.ckpt_path,map_location=device))
        saliency_model.eval()
    to_tensor=ToTensor()
    to_pil=ToPILImage()
    imgs=torch.stack([to_tensor(Image.open(f)) for f in img_list]).to(device)
    result = saliency_model(imgs)
    if mutliply:
        result=result*imgs
    for array,save_name in zip(result,output_list):
        to_pil(array.cpu()).save(save_name)




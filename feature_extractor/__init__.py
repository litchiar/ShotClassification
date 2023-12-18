import os
from .get_flow import get_flow_
from .get_segmentation import get_segmentation_, get_segmentation_single_image_
from .get_saliency import get_saliency_

def get_flow(video_path, output_dir, clip_N):
    get_flow_(video_path, output_dir, clip_N)


def get_segmentation(img_file, output_dir=None, index_offset=0):
    get_segmentation_(img_file, output_dir, offset=index_offset)
    pass


def get_segmentation_single_image(img_file, output_dir=None):
    get_segmentation_single_image_(img_file, output_dir)


def get_saliency(img_list,output_list,device='cuda:0',mutliply=False,invert=False):
    get_saliency_(img_list,output_list,device=device,mutliply=mutliply,invert=invert)
    pass



# 三个方法
import os
from .get_flow import get_flow_
from .get_segmentation import get_segmentation_, get_segmentation_single_image_
from .get_saliency import get_saliency_

# 1.输入视频文件将一个镜头视频文件分为N段，输出N张图片和光流
def get_flow(video_path, output_dir, clip_N):
    get_flow_(video_path, output_dir, clip_N)


# 2.获取图像语义分割结果
def get_segmentation(img_file, output_dir=None, index_offset=0):
    get_segmentation_(img_file, output_dir, offset=index_offset)
    pass


def get_segmentation_single_image(img_file, output_dir=None):
    get_segmentation_single_image_(img_file, output_dir)


# 3.获取图像显著度预测结果,尚未实现
def get_saliency(img_list,output_list,device='cuda:0',mutliply=False,invert=False):
    get_saliency_(img_list,output_list,device=device,mutliply=mutliply,invert=invert)
    pass



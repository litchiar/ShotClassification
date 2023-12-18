import math
import os
import copy
from typing import List, Optional, Sequence, Union
import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmflow.models import build_flow_estimator
from matplotlib import pyplot as plt
from mmflow.datasets.pipelines import Compose
from mmcv.ops import Correlation
from PIL import Image
dirname, filename = os.path.split(os.path.abspath(__file__))
mmflow_dir = os.path.join(dirname,'mmflow_file')
for f in os.listdir(mmflow_dir):
    if f.endswith('.pth'):
        FLOW_MODEL = os.path.join(mmflow_dir, f)
    if f.endswith('.py'):
        FLOW_CONFIG = os.path.join(mmflow_dir, f)
assert FLOW_MODEL is not None
assert FLOW_CONFIG is not None

flownet_model=None

def init_model(config: Union[str, mmcv.Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None) -> torch.nn.Module:
    """Initialize a flow estimator from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file output_dir or the config
            object.
        checkpoint (str, optional): Checkpoint output_dir. If left as None, the model
            will not load any weights. Default to: None.
        device (str): Represent the device. Default to: 'cuda:0'.
        cfg_options (dict, optional): Options to override some settings in the
            used config. Default to: None.
    Returns:
        nn.Module: The constructed flow estimator.
    """

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    config.model.train_cfg = None
    model = build_flow_estimator(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model

def inference_model(
        model: torch.nn.Module,
        img1s: Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]],
        img2s: Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]],
        valids: Optional[Union[str, np.ndarray, Sequence[str],
                               Sequence[np.ndarray]]] = None
) -> Union[List[np.ndarray], np.ndarray]:
    """Inference images pairs with the flow estimator.

    Args:
        model (nn.Module): The loaded flow estimator.
        img1s (str/ndarray or sequence[str/ndarray]): Either image files or
            loaded images.
        img2s (str/ndarray or sequence[str/ndarray]): Either image files or
            loaded images.
        valids (str/ndarray or list[str/ndarray], optional): Either mask files
            or loaded mask. If the predicted flow is sparse, valid mask will
            filter the output flow map.
    Returns:
        If img-pairs is a list or tuple, the same length list type results
        will be returned, otherwise return the flow map from image1 to image2
        directly.
    """
    if isinstance(img1s, (list, tuple)):
        is_batch = True
    else:
        img1s = [img1s]
        img2s = [img2s]
        valids = [valids]
        is_batch = False
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if cfg.data.test.type == 'ConcatDataset':
        cfg = copy.deepcopy(cfg.data.test.datasets[0])
    else:
        cfg = copy.deepcopy(cfg.data.test)

    if isinstance(img1s[0], np.ndarray):
        # set loading pipeline type
        cfg.pipeline[0].type = 'LoadImageFromWebcam'

    # as load annotation is for online evaluation
    # there is no need to load annotation.
    if dict(type='LoadAnnotations') in cfg.pipeline:
        cfg.pipeline.remove(dict(type='LoadAnnotations'))
    if dict(type='LoadAnnotations', sparse=True) in cfg.pipeline:
        cfg.pipeline.remove(dict(type='LoadAnnotations', sparse=True))

    if 'flow_gt' in cfg.pipeline[-1]['meta_keys']:
        cfg.pipeline[-1]['meta_keys'].remove('flow_gt')
    if 'flow_fw_gt' in cfg.pipeline[-1]['meta_keys']:
        cfg.pipeline[-1]['meta_keys'].remove('flow_fw_gt')
    if 'flow_bw_gt' in cfg.pipeline[-1]['meta_keys']:
        cfg.pipeline[-1]['meta_keys'].remove('flow_bw_gt')

    test_pipeline = Compose(cfg.pipeline)
    datas = []
    valid_masks = []
    for img1, img2, valid in zip(img1s, img2s, valids):
        # prepare data
        if isinstance(valid, str):
            # there is no real example to test the function for loading valid
            # masks
            valid = mmcv.imread(valid, flag='grayscale')
        if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
            # directly add img and valid mask
            data = dict(img1=img1, img2=img2, valid=valid)
        else:
            # add information into dict
            data = dict(
                img_info=dict(filename1=img1, filename2=img2),
                img1_prefix=None,
                img2_prefix=None,
                valid=valid)
        data['img_fields'] = ['img1', 'img2']
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
        valid_masks.append(valid)

    data = collate(datas, samples_per_gpu=len(img1s))
    # just get the actual data from DataContainer

    data['img_metas'] = data['img_metas'].data[0]
    data['imgs'] = data['imgs'].data[0]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, Correlation
            ), 'CPU inference with Correlation is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(test_mode=True, **data)

    if valid_masks[0] is not None:
        # filter the output flow map
        for result, valid in zip(results, valid_masks):
            if result.get('flow', None) is not None:
                result['flow'] *= valid
            elif result.get('flow_fw', None) is not None:
                result['flow_fw'] *= valid

    if not is_batch:
        # only can inference flow of forward direction
        if results[0].get('flow', None) is not None:
            return results[0]['flow']
        if results[0].get('flow_fw', None) is not None:
            return results[0]['flow_fw']
    else:
        return results


def visualize_flow(flow: np.ndarray, save_file: str = None) -> np.ndarray:
    """Flow visualization function.

    Args:
        flow (ndarray): The flow will be render
        save_dir ([type], optional): save dir. Defaults to None.
    Returns:
        ndarray: flow map image with RGB order.
    """

    # return value from mmcv.flow2rgb is [0, 1.] with type np.float32
    flow_map = np.uint8(mmcv.flow2rgb(flow) * 255.)
    return flow_map


def process_video(video_path, model, output_dir, seg_count=8, pic_stg='middle',default_size=(192*2,108*2),extract_flow_value=False,**kwargs):#extract_flow_value 临时改为True，一般设置为False
    assert pic_stg in ['first', 'middle', 'last']
    cap = cv2.VideoCapture(video_path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(7))
    if frame_count <= seg_count:
        # print(video_path,'frame_count <= seg_count')
        return False
    # 一个视频分成seg段，然后得到seg张光流的图片加上seg张光流
    average_duration = frame_count / seg_count
    video_clip = [[round(i * average_duration), round(i * average_duration + average_duration - 1)] for i in
                  range(seg_count)]
    video_clip[-1][-1] = frame_count - 1
    if extract_flow_value:
        flow_results=[]
    for i, clip in enumerate(video_clip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip[0])
        res, img_first = cap.read()
        img_first = cv2.cvtColor(img_first, cv2.COLOR_RGB2BGR)
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip[1])
        res, img_last = cap.read()
        img_last = cv2.cvtColor(img_last, cv2.COLOR_RGB2BGR)
        result = inference_model(model, img_first, img_last)
        flow_map = visualize_flow(result, None)
        if extract_flow_value:
            flow_results.append(result.mean(axis=(0,1)))
            pass
        Image.fromarray(cv2.resize(flow_map,default_size, interpolation = cv2.INTER_AREA)).save(os.path.join(output_dir, f"flow_{i}.jpg"))
        if pic_stg == 'first':
            Image.fromarray(img_first).save(os.path.join(output_dir, f"image_{i}.jpg"))
            pass
        elif pic_stg == 'last':
            Image.fromarray(img_last).save(os.path.join(output_dir, f"image_{i}.jpg"))
            # origin.append(img_last)
            pass
        else:
            pass
            cap.set(cv2.CAP_PROP_POS_FRAMES, (clip[0] + clip[1]) // 2)
            res, img_middle = cap.read()
            Image.fromarray(cv2.cvtColor(cv2.resize(img_middle,default_size, interpolation = cv2.INTER_AREA), cv2.COLOR_RGB2BGR)).save(os.path.join(output_dir, f"image_{i}.jpg"))
            # origin.append(cv2.cvtColor(img_middle, cv2.COLOR_RGB2BGR))
    if extract_flow_value:
        flow_result=np.stack(flow_results)
        np.save(os.path.join(output_dir,os.path.basename(output_dir)+'.npy'),flow_result)
    cap.release()
    return True


def get_flow_(video_path,output_dir,clip_N,**kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    global flownet_model
    if flownet_model is None:
        flownet_model=init_model(FLOW_CONFIG,FLOW_MODEL)
    return process_video(video_path,flownet_model,output_dir,seg_count=clip_N,**kwargs)

def extract_flow_per_second(video_path,output_path):
    if not os.path.exists(video_path):
        return
    #output_path=os.path.join(os.getcwd(),'temp')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cap = cv2.VideoCapture(video_path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(7))
    seconds=math.floor(frame_count/fps)
    print(f'seconds:{math.floor(frame_count/fps)}')
    from tqdm import tqdm
    flows=[]
    for i in tqdm(range(seconds)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i*fps))
        res, img_start = cap.read()
        img_start = cv2.cvtColor(img_start, cv2.COLOR_RGB2BGR)
        cap.set(cv2.CAP_PROP_POS_FRAMES,int((i+1)*fps))
        res, img_last = cap.read()
        img_last = cv2.cvtColor(img_last, cv2.COLOR_RGB2BGR)
        global flownet_model
        if flownet_model is None:
            flownet_model = init_model(FLOW_CONFIG, FLOW_MODEL)
        result = inference_model(flownet_model, img_start, img_last)

        #mean_flow=np.mean(result,axis=(0,1))
        #flows.append(mean_flow)
        Image.fromarray(img_start).save(os.path.join(output_path, f"image_{i}.jpg"))
        flow_map = visualize_flow(result, None)
        Image.fromarray(flow_map).save( os.path.join(output_path, f"flow_{i}.jpg"))

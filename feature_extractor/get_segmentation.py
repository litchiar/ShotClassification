import os
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
dirname, filename = os.path.split(os.path.abspath(__file__))
mmseg_dir = os.path.join(dirname, 'mmseg_file')
for f in os.listdir(mmseg_dir):
    if f.endswith('.pth'):
        SEG_MODEL = os.path.join(mmseg_dir, f)
    if f.endswith('.py'):
        SEG_CONFIG = os.path.join(mmseg_dir, f)
#assert SEG_MODEL is not None
#assert SEG_CONFIG is not None
psp_model = None

def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def inference_segmentor(model, imgs):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    for img in imgs:
        img_data = dict(img=img)
        img_data = test_pipeline(img_data)
        data.append(img_data)
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def save_result(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True,
                       out_file=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

def get_segmentation_(img_files, output_dir=None,offset=0):#
    if not os.path.exists(img_files[0]):
        return
    if len(img_files)<=0:
        return
    if output_dir is None:
        output_dir, _ = os.path.split(os.path.abspath(img_files[0]))
    global psp_model
    if psp_model is None:
        psp_model=init_segmentor(SEG_CONFIG,SEG_MODEL)
    result =inference_segmentor(psp_model,img_files)
    for i,r in enumerate(result):
        save_result(psp_model,img_files[i],[r],None,opacity=1,out_file=os.path.join(output_dir,f"seg_{i+offset}.jpg"))

def get_segmentation_single_image_(img_file, output_dir=None):
    img_files=[img_file]
    if not os.path.exists(img_files[0]):
        return
    if len(img_files) <= 0:
        return
    if output_dir is None:
        output_dir, _ = os.path.split(os.path.abspath(img_files[0]))
    global psp_model
    if psp_model is None:
        psp_model = init_segmentor(SEG_CONFIG, SEG_MODEL)
    result = inference_segmentor(psp_model, img_files)
    for i, r in enumerate(result):
        save_result(psp_model, img_files[i], [r], None, opacity=1,
                    out_file=os.path.join(output_dir, os.path.basename(img_file).split('.')[0]+"_seg.jpg"))



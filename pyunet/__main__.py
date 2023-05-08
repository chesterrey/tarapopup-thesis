import sys
import argparse
import os
import torch
import datetime

from .modules.train import Train
from .modules.forward import Forward
from .modules.monitor import Monitor
from .modules.monitor_onnx import MonitorOnnx
from .modules.export_onnx import ExportOnnx
from .modules.generate_tiff import GenerateTiff
from .modules.sample_pair import SamplePair
from .modules.generate_dataset import GenerateDataset
from .modules.sample_frame import SampleFrame
from .modules.extract_unique_gray import ExtractUniqueGray
from .modules.benchmark import Benchmark

mode_choices = [
    "train",
    "forward",
    "generate-tiff",
    "monitor",
    "monitor-onnx",
    "export-onnx",
    "sample-pair",
    "benchmark",
    "generate-dataset",
    "sample-frame",
    "extract-unique-gray"
]

model_type_choices = [
    "unet",
    "unet_attn",
    "unet_attn_dp",
    "wnet",
]

default_dataset_name = (datetime.datetime.now()).strftime("%Y%m%d%H%M%S")

def main():
    parser = argparse.ArgumentParser(description="PyUNET: Python implementation of UNET")

    parser.add_argument("--mode", help="Mode to be used", choices=mode_choices, type=str, required=True)
    parser.add_argument("--img-width", help="Image width", type=int, default=256)
    parser.add_argument("--img-height", help="Image height", type=int, default=256)
    parser.add_argument("--device", help="Device used for training", choices=["cpu", "cuda"], type=str, default="cpu")
    parser.add_argument("--gpu-index", help="GPU index", type=int, default=0)
    parser.add_argument("--input-img-dir", help="Input image directory", type=str)
    parser.add_argument("--output-img-dir", help="Output image directory", type=str)
    parser.add_argument("--input-mask-dir", help="Input mask directory", type=str)
    parser.add_argument("--epochs", help="Epoch count", type=int, default=100)
    parser.add_argument("--learning-rate", help="Learning rate", type=float, default=0.001)
    parser.add_argument("--model-file", help="Model file", type=str, default="model.pth")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=1)
    parser.add_argument("--input-img", help="Input image", type=str, required=False)
    parser.add_argument("--in-channels", help="In Channels", type=int, default=3)
    parser.add_argument("--out-channels", help="Out Channels", type=int, default=2)
    parser.add_argument("--unique-values", help="Features", type=str, nargs='+', required=False)
    parser.add_argument("--video", help="Video index", type=str, default="0")
    parser.add_argument("--img-suffix", help="Img Suffix", type=str, default="jpg")
    parser.add_argument("--cont", help="Continue training", type=bool, default=False)
    parser.add_argument("--loss-type", help="Type of loss function", type=str, default='CE')
    parser.add_argument("--display-width", help="Display width", type=int, default=800)
    parser.add_argument("--display-height", help="Display height", type=int, default=640)
    parser.add_argument("--model-type", help="UNet model type", type=str, choices=model_type_choices, default='unet')
    parser.add_argument("--dataset-name", help="Dataset name", type=str, default="data-{}".format(default_dataset_name))
    parser.add_argument("--test-img-dir", help="Test image dir", type=str, default="test/images")
    parser.add_argument("--test-mask-dir", help="Test mask dir", type=str, default="test/masks")
    parser.add_argument("--sampled-index", help="Sampled index", type=int, default=-1)
    parser.add_argument("--export-file", help="Export file", type=str, default='model.onnx')

    args = parser.parse_args()

    mode            = args.mode
    img_width       = args.img_width
    img_height      = args.img_height
    device          = args.device
    gpu_index       = args.gpu_index
    input_img_dir   = args.input_img_dir
    output_img_dir  = args.output_img_dir
    input_mask_dir  = args.input_mask_dir
    epochs          = args.epochs
    learning_rate   = args.learning_rate
    model_file      = args.model_file
    batch_size      = args.batch_size
    input_img       = args.input_img
    in_channels     = args.in_channels
    out_channels    = args.out_channels
    unique_values   = args.unique_values
    video           = args.video
    cont            = args.cont
    loss_type       = args.loss_type
    display_width   = args.display_width
    display_height  = args.display_height
    model_type      = args.model_type
    img_suffix      = args.img_suffix
    dataset_name    = args.dataset_name
    test_img_dir    = args.test_img_dir
    test_mask_dir   = args.test_mask_dir
    sampled_index   = args.sampled_index
    export_file     = args.export_file

    if mode =="train":
        params = {
            'img_width':        img_width,
            'img_height':       img_height,
            'device':           device,
            'gpu_index':        gpu_index,
            'input_img_dir':    input_img_dir,
            'input_mask_dir':   input_mask_dir,
            'epochs':           epochs,
            'learning_rate':    learning_rate,
            'model_file':       model_file,
            'batch_size':       batch_size,
            'in_channels':      in_channels,
            'out_channels':     out_channels,
            'cont':             cont,
            'loss_type':        loss_type,
            'model_type':       model_type
        }

        cmd = Train(params=params)
        cmd.execute()

    elif mode =="forward":
        params = {
            'model_file':   model_file,
            'img_width':    img_width,
            'img_height':   img_height,
            'input_img':    input_img,
            'gpu_index':    gpu_index,
            'device':       device,
            'model_type':   model_type
        }

        cmd = Forward(params=params)
        cmd.execute()

    elif mode =="monitor":
        params = {
            'model_file':       model_file,
            'img_width':        img_width,
            'img_height':       img_height,
            'video':            video,
            'gpu_index':        gpu_index,
            'device':           device,
            'in_channels':      in_channels,
            'out_channels':     out_channels,
            'display_hidth':    display_width,
            'display_height':   display_height,
            'model_type':       model_type
        }

        cmd = Monitor(params=params)
        cmd.execute()

    elif mode == "monitor-onnx":
        params = {
            'model_file':       model_file,
            'img_width':        img_width,
            'img_height':       img_height,
            'video':            video,
            'gpu_index':        gpu_index,
            'device':           device,
            'in_channels':      in_channels,
            'out_channels':     out_channels,
            'display_hidth':    display_width,
            'display_height':   display_height,
            'model_type':       model_type
        }

        cmd = MonitorOnnx(params=params)
        cmd.execute()

    elif mode == "sample-pair":
        params = {
            'img_width':        img_width,
            'img_height':       img_height,
            'input_img_dir':    input_img_dir,
            'input_mask_dir':   input_mask_dir,
            'model_file':       model_file,
            'model_type':       model_type,
            'device':           device,
            'gpu_index':        gpu_index,
            'sampled_index':    sampled_index
        }
        
        cmd = SamplePair(params=params)
        cmd.execute()

    elif mode == "generate-tiff":
        params = {
            'input_img_dir':    input_img_dir,
            'output_img_dir':   output_img_dir,
            'unique_values':    unique_values,
            'img_suffix':       img_suffix
        }

        cmd = GenerateTiff(params=params)
        cmd.execute()
    
    elif mode == "generate-dataset":
        params = {
            'dataset_name':     dataset_name,
            'input_img_dir':    input_img_dir,
            'input_mask_dir':   input_mask_dir,
        }

        cmd = GenerateDataset(params=params)
        cmd.execute()

    elif mode == "sample-frame":
        params = {
            'img_width':    img_width,
            'img_height':   img_height,
            'input_img':    input_img,
            'model_type':   model_type,
            'model_file':   model_file,
            'in_channels':  in_channels,
            'out_channels': out_channels,
            'device':       device
        }

        cmd = SampleFrame(params=params)
        cmd.execute()

    elif mode == "extract-unique-gray":
        params = {
            "input_img_dir":    input_img_dir,
            "img_suffix":       img_suffix
        }

        cmd = ExtractUniqueGray(params=params)
        cmd.execute()

    elif mode == "benchmark":
        params = {
            'img_width':        img_width,
            'img_height':       img_height,
            'device':           device,
            'gpu_index':        gpu_index,
            'input_img_dir':    input_img_dir,
            'input_mask_dir':   input_mask_dir,
            'model_file':       model_file,
            'model_type':       model_type,
            'in_channels':      in_channels,
            'out_channels':     out_channels,
        }

        cmd = Benchmark(params=params)
        cmd.execute()

    elif mode == "export-onnx":
        params = {
            'device':       device,
            'gpu_index':    gpu_index,
            'model_file':   model_file,
            'img_width':    img_width,
            'img_height':   img_height,
            'in_channels':  in_channels,
            'out_channels': out_channels,
            'model_type':   model_type,
            'export_file':  export_file
        }

        cmd = ExportOnnx(params=params)
        cmd.execute()

    else:
        raise ValueError("Invalid mode {}".format(mode))

if __name__ == '__main__':
    main()

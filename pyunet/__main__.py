import sys
import argparse
import os
import datetime
import os.path
import sys
import json
import optuna
from optuna.pruners import MedianPruner
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.train import Train
from modules.forward import Forward
from modules.monitor import Monitor
from modules.monitor_onnx import MonitorOnnx
from modules.export_onnx import ExportOnnx
from modules.generate_tiff import GenerateTiff
from modules.sample_pair import SamplePair
from modules.depth.sample_pair_depth import SamplePairDepth
from modules.generate_dataset import GenerateDataset
from modules.sample_frame import SampleFrame
from modules.extract_unique_gray import ExtractUniqueGray
from modules.benchmark import Benchmark
from modules.assert_model import AssertModel
from modules.rgb2mask import Rgb2Mask

# Depth
from modules.depth.train_depth import TrainDepth

from lib.helpers.extract_params_from_config import ExtractParamsFromConfig

mode_choices = [
    "train",
    "train-depth",
    "forward",
    "generate-tiff",
    "monitor",
    "monitor-onnx",
    "export-onnx",
    "sample-pair",
    "sample-pair-depth",
    "benchmark",
    "generate-dataset",
    "sample-frame",
    "extract-unique-gray",
    "assert-model",
    "rgb2mask",
    "train-depth",
    "optimize-hyperparameters"
]

model_type_choices = [
    "unet",
    "unet_attn",
    "unet_attn_dp",
    "unet_attn_ghost",
    "unet_attn_inverted_residual_block",
    "unet_attn_stacked_ghost_irb",
    "unet_depth",
    "unet_attn_depth",
    "unet_attn_dp_depth",
    "wnet",
    "vgg_unet",
    "wnet_vggunet_vgg",
    "wnet_vggunet_vggunet",
    "wnet_vggunet_unet",
    "wnet_unet_vggunet",
    "double_unet",
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
    parser.add_argument("--unique-values", help="Features", type=int, nargs='+', required=False)
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
    parser.add_argument("--image-file", help="Image file", type=str, default="img.png")
    parser.add_argument("--config-file", help="Config file", type=str, default="")

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
    image_file      = args.image_file
    config_file     = args.config_file

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

        if config_file:
            with open(config_file) as json_file:
                params = json.load(json_file)

        cmd = Train(params=params)
        cmd.execute()
    
    elif mode == "optimize-hyperparameters":

        # Define the hyperparameter search space for your U-Net model
        def objective(trial):
            params = {
                'img_width': args.img_width,
                'img_height': args.img_height,
                'device': args.device,
                'gpu_index': args.gpu_index,
                'input_img_dir': args.input_img_dir,
                'input_mask_dir': args.input_mask_dir,
                'epochs': args.epochs,
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                'model_file': args.model_file,
                'batch_size': args.batch_size,
                'in_channels': args.in_channels,
                'out_channels': args.out_channels,
                'cont': args.cont,
                'loss_type': args.loss_type,
                'model_type': args.model_type,
            }

            if config_file:
                with open(config_file) as json_file:
                    params = json.load(json_file)

            # Create a Train instance with the sampled hyperparameters
            trainer = Train(params=params)

            # Execute the training with the sampled hyperparameters
            trainer.execute()

            # Return the validation loss as the result of the objective function
            return trainer.accuracies[-1]

        # Create an Optuna study for hyperparameter optimization
        study = optuna.create_study(direction='maximize', pruner='tpe')
        study.optimize(objective, n_trials=3)  # You can adjust the number of trials

        # Access the best hyperparameters and their corresponding accuracy
        best_params = study.best_params
        best_accuracy = study.best_value

        print("Best Hyperparameters:")
        print(best_params)
        print(f"Best Accuracy: {best_accuracy:.4f}")

    elif mode == "train-depth":
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
            'model_type':       model_type
        }

        if config_file:
            with open(config_file) as json_file:
                params = json.load(json_file)

        cmd = TrainDepth(params=params)
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
            'display_width':    display_width,
            'display_height':   display_height,
            'model_type':       model_type
        }

        if config_file:
            with open(config_file) as json_file:
                params = json.load(json_file)

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

    # Exclusively run mode with json config
    elif mode == "sample-pair":
        with open(config_file) as json_file:
            params = json.load(json_file)
        
        cmd = SamplePair(params=params)
        cmd.execute()

    elif mode == "sample-pair-depth":
        with open(config_file) as json_file:
            params = json.load(json_file)
        
        cmd = SamplePairDepth(params=params)
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
        with open(config_file) as json_file:
            params = json.load(json_file)

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

    elif mode == "assert-model":
        params = {
            'device':       device,
            'gpu_index':    gpu_index,
            'model_type':   model_type,
            'img_width':    img_width,
            'img_height':   img_height,
            'in_channels':  in_channels,
            'out_channels': out_channels
        }

        cmd = AssertModel(params=params)
        cmd.execute()
    
    elif mode == "rgb2mask":
        params = {
            'img_width':    img_width,
            'img_height':   img_height,
            'config_file':  config_file,
            'image_file':   image_file
        }

        cmd = Rgb2Mask(params=params)
        cmd.execute()

    else:
        raise ValueError("Invalid mode {}".format(mode))

if __name__ == '__main__':
    main()

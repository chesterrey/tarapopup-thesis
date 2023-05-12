set DIR="C:\Users\acer\Documents\pyunet"

cd %DIR%

@REM Parameters
set DEVICE=cuda
set MODEL_TYPE=wnet
set GPU_INDEX=0
@REM set INPUT_IMG="C:\Users\acer\Documents\pyunet\notebooks\images\covid19ctscan\images\scan_slice00.png"
@REM set INPUT_IMG="C:\Users\acer\Documents\pyunet\dataset\combined\train_images\131368cc17e44240_28955.jpg"
set INPUT_IMG="D:\School Stuff\CSCI 199.1\chapter 3\pyunet\data\train_images\131368cc17e44240_28955.jpg"
set MODEL_FILE="C:\Users\acer\Documents\pyunet\model.pth"

python -m pyunet^
  --mode forward^
  --model-type %MODEL_TYPE%^
  --model-file %MODEL_FILE%^
  --input-img %INPUT_IMG%^
  --device %DEVICE%^
  --gpu-index %GPU_INDEX% 
  


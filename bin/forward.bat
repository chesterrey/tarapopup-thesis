set DIR="C:\Users\acer\Documents\pyunet"

cd %DIR%

@REM Parameters
set DEVICE=cuda
set MODEL_TYPE=wnet
set GPU_INDEX=0
@REM set INPUT_IMG="C:\Users\acer\Documents\pyunet\notebooks\images\covid19ctscan\images\scan_slice00.png"
@REM set INPUT_IMG="C:\Users\acer\Documents\pyunet\dataset\combined\train_images\131368cc17e44240_28955.jpg"
set INPUT_IMG="C:\Users\acer\Documents\pyunet\notebooks\images\ebhi-seg-normal\images\gt2000000-1-400-001.png"
@REM set INPUT_IMG="C:\Users\acer\Documents\pyunet\dataset\combined\val_images\bone (2).JPG"
set MODEL_FILE="C:\Users\acer\Documents\pyunet\model.pth"

python -m pyunet^
  --mode forward^
  --model-type %MODEL_TYPE%^
  --model-file %MODEL_FILE%^
  --input-img %INPUT_IMG%^
  --device %DEVICE%^
  --gpu-index %GPU_INDEX% 
  


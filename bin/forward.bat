set DIR="C:\Users\acer\Documents\pyunet"

cd %DIR%

@REM Parameters
set DEVICE=cuda
set MODEL_TYPE=wnet
set GPU_INDEX=0
set INPUT_IMG="C:\Users\acer\Documents\pyunet\notebooks\images\covid19ctscan\images\scan_slice00.png"
set MODEL_FILE="C:\Users\acer\Documents\pyunet\model.pth"

python -m pyunet^
  --mode forward^
  --model-type %MODEL_TYPE%^
  --model-file %MODEL_FILE%^
  --input-img %INPUT_IMG%^
  --device %DEVICE%^
  --gpu-index %GPU_INDEX% 
  


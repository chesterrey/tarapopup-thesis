set DIR="C:\Users\acer\Documents\pyunet"

cd %DIR%

@REM Parameters
set DEVICE=cuda
set MODEL_TYPE=wnet
set GPU_INDEX=0
set IN_CHANNELS=3
set OUT_CHANNELS=3
set CLASSES=2
@REM set INPUT_IMG="C:\Users\acer\Documents\pyunet\notebooks\images\covid19ctscan\images\scan_slice00.png"
@REM set INPUT_IMG="C:\Users\acer\Documents\pyunet\dataset\combined\train_images\131368cc17e44240_28955.jpg"
@REM set INPUT_IMG="C:\Users\acer\Documents\pyunet\notebooks\images\ebhi-seg-normal\images\gt2000000-1-400-001.png"
set INPUT_IMG="C:\Users\acer\Documents\pyunet\dataset\combined\val_images\bone (1).JPG"
set MODEL_FILE="C:\Users\acer\Documents\pyunet\models\wnet\wnet_100.pth"

python -m pyunet^
  --mode forward^
  --model-type %MODEL_TYPE%^
  --model-file %MODEL_FILE%^
  --input-img %INPUT_IMG%^
  --device %DEVICE%^
  --gpu-index %GPU_INDEX%^
  --in-channels %IN_CHANNELS%^
  --out-channels %OUT_CHANNELS%^
  --classes %CLASSES%
  


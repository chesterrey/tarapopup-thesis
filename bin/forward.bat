set DIR="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis"

cd %DIR%

@REM Parameters
set DEVICE=cuda
set MODEL_TYPE=wnet
set GPU_INDEX=0
set INPUT_IMG="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis\dataset\combined\train_images\bone (3).JPG"
set MODEL_FILE="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis\model.pth"

python -m pyunet^
  --mode forward^
  --model-type %MODEL_TYPE%^
  --model-file %MODEL_FILE%^
  --input-img %INPUT_IMG%^
  --device %DEVICE%^
  --gpu-index %GPU_INDEX% 
  


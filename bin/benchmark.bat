set DIR="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis"
cd %DIR%

@REM Parameters:
set DEVICE=cuda
set GPU_INDEX=0
set IMG_WIDTH=512
set IMG_HEIGHT=512
set INPUT_IMG_DIR="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis\dataset\combined\val_images"
set MASKED_IMG_DIR="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis\dataset\combined\val_masks"
set MODEL_FILE="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis\model.pth"
set MODEL_TYPE=wnet


python -m pyunet^
 --mode benchmark^
 --img-width %IMG_WIDTH%^
 --img-height %IMG_HEIGHT%^
 --device %DEVICE%^
 --gpu-index %GPU_INDEX%^
 --input-img-dir %INPUT_IMG_DIR%^
 --input-mask-dir %MASKED_IMG_DIR%^
 --model-type %MODEL_TYPE%^
 --model-file %MODEL_FILE%
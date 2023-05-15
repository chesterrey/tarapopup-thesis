set DIR="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis"
cd %DIR%

@REM Parameters:
set DEVICE=cuda
set GPU_INDEX=0
set IMG_WIDTH=512
set IMG_HEIGHT=512
set INPUT_IMG_DIR="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis\dataset\combined\train_images"
set MASKED_IMG_DIR="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis\dataset\combined\train_masks"
set MODEL_FILE="C:\Users\Andre\OneDrive - ateneo.edu\Documents\vscode_projects\tarapopup-thesis\model.pth"
set MODEL_TYPE=wnet

python -m pyunet^
 --mode sample-pair^
 --img-width %IMG_WIDTH%^
 --img-height %IMG_HEIGHT%^
 --input-img-dir %INPUT_IMG_DIR%^
 --input-mask-dir %MASKED_IMG_DIR%^
 --model-file %MODEL_FILE%^
 --device %DEVICE%^
 --model-type %MODEL_TYPE%^
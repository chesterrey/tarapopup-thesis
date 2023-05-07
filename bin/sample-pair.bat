set DIR="C:\Users\acer\Documents\pyunet"
cd %DIR%

@REM Parameters:
set DEVICE=cuda
set GPU_INDEX=0
set IMG_WIDTH=336
set IMG_HEIGHT=336
set INPUT_IMG_DIR="D:\School Stuff\CSCI 199.1\chapter 3\pyunet\data\train_images"
set MASKED_IMG_DIR="D:\School Stuff\CSCI 199.1\chapter 3\pyunet\data\train_masks"
set MODEL_FILE="C:\Users\acer\Documents\pyunet\model.pth"
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
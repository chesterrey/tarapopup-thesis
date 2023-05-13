set DIR="C:\Users\acer\Documents\pyunet"
cd %DIR%

@REM Parameters:
set DEVICE=cuda
set GPU_INDEX=0
set IMG_WIDTH=336
set IMG_HEIGHT=336
set INPUT_IMG_DIR="C:\Users\acer\Documents\pyunet\dataset\combined\train_images"
set MASKED_IMG_DIR="C:\Users\acer\Documents\pyunet\dataset\combined\train_masks"
@REM set INPUT_IMG_DIR="C:\Users\acer\Documents\pyunet\notebooks\images\ebhi-seg-normal\images"
@REM set MASKED_IMG_DIR="C:\Users\acer\Documents\pyunet\notebooks\images\ebhi-seg-normal\masks"
set MODEL_FILE="C:\Users\acer\Documents\pyunet\model.pth"
set BATCH_SIZE=1
set EPOCHS=100
set LEARNING_RATE=0.003
set IN_CHANNELS=3
set OUT_CHANNELS=2
set LOSS_TYPE=CE
set MODEL_TYPE=wnet
set CONT=False

 python -m pyunet^
  --mode train^
  --device %DEVICE%^
  --gpu-index %GPU_INDEX%^
  --img-width %IMG_WIDTH%^
  --img-height %IMG_HEIGHT%^
  --input-img-dir %INPUT_IMG_DIR%^
  --input-mask-dir %MASKED_IMG_DIR%^
  --epochs %EPOCHS%^
  --batch-size %BATCH_SIZE%^
  --learning-rate %LEARNING_RATE%^
  --model-type %MODEL_TYPE%^
  --loss-type %LOSS_TYPE%^
  --cont %CONT%
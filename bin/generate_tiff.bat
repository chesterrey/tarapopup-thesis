set DIR="C:/Users/acer/Documents/pyunet"

cd %DIR%

set UNIQUE_VALUES="0 255"
set INPUT_IMG_DIR="C:\Users\acer\Documents\pyunet\dataset\combined\not_preprocessed_mask_rbg"
set OUTPUT_IMG_DIR="C:\Users\acer\Documents\pyunet\dataset\combined\train_masks"
set IMG_SUFFIX=png

python -m pyunet^
  --mode generate-tiff^
  --unique-values %UNIQUE_VALUES%^
  --input-img-dir %INPUT_IMG_DIR%^
  --output-img-dir %OUTPUT_IMG_DIR%^
  --img-suffix %IMG_SUFFIX%

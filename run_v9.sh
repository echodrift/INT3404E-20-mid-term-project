# this shell script is used to execute yolov9 (fork from https://github.com/WongKinYiu/yolov9)
# specify path to `train_dual.py` and your `data.yaml`

python train_dual.py \
--workers 8 \
--device 0 \
--sync-bn \
--batch 2 \
--data '/kaggle/working/image_processing/datasets/wb_localization_dataset/training.yaml' \
--img 928 \
--cfg '/kaggle/working/image_processing/yolov9/models/detect/yolov9-e.yaml' \
--weights '/kaggle/working/image_processing/yolov9-e-converted.pt' \
--name 'yolov9-e' \
--hyp '/kaggle/working/image_processing/yolov9/data/hyps/hyp.scratch-high.yaml' \
--min-items 0 \
--epochs 100 \
--close-mosaic 15
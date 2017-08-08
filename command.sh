'''
This file is used to give example of command line of training and testing.
'''


# =============== TRAIN ===============
python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB_combined_data" --combine_dataset=True
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet_combined_data" --combine_dataset=True
python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB"

# CamVid
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet"
# Cityscapes
python train_enet.py --weighting="ENET" --num_epochs=300 --num_classes=20 --image_height=256 image_width=512 --dataset="Cityscapes" --dataset_dir="./cityscapes" --logdir="./log/train_original_ENet"


# =============== TEST ===============
python test_enet.py --checkpoint_dir="./log/train_original_MFB" --logdir="./log/test_original_MFB"
python test_enet.py --checkpoint_dir="./log/train_original_ENet_combined_data" --logdir="./log/test_original_ENet_combined_data"
python test_enet.py --checkpoint_dir="./log/train_original_MFB_combined_data" --logdir="./log/test_original_MFB_combined_data"

# CamVid
python test_enet.py --checkpoint_dir="./log/train_original_ENet" --logdir="./log/test_original_ENet"
# Cityscapes
python test_enet.py --num_classes=20 --image_height=1024 image_width=2048 --dataset="Cityscapes" --dataset_dir="./cityscapes" --checkpoint_dir="./log/train_original_ENet" --logdir="./log/test_original_ENet"


# =============== DEMO ===============
# CamVid
python demo_enet.py --checkpoint_dir="./log/train_original_ENet" --logdir="./log/demo_original_ENet"
# Cityscapes
python demo_enet.py --num_classes=20 --image_height=1024 --image_width=2048 --dataset="Cityscapes" --dataset_dir="./cityscapes" --checkpoint_dir="./log/train_original_ENet" --logdir="./log/demo_original_ENet"

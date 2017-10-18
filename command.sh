'''
This file is used to give example of command line of training and testing.
'''


# =============== TRAIN ===============
python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="../log/train_original_MFB_combined_data" --combine_dataset=True
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="../log/train_original_ENet_combined_data" --combine_dataset=True
python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="../log/train_original_MFB"

# CamVid
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="../log/train_original_ENet_Cam"
# Cityscapes
python train_enet.py --weighting="ENET" --num_epochs=50 --num_classes=20 --image_height=1024 --image_width=2048 --dataset="Cityscapes" --logdir="../log/train_original_ENet_City"
#NYU
python train_enet.py --weighting="ENET" --num_epochs=500 --num_classes=5 --image_height=480 --image_width=640 --dataset="NYU" --logdir="../log/train_original_ENet_NYU"
#ADE
python train_enet.py --weighting="ENET" --num_epochs=40 --num_classes=27 --image_height=480 --image_width=640 --dataset="ADE" --logdir="../log/train_original_ENet_ADE"


# =============== TEST ===============
python test_enet.py --checkpoint_dir="../log/train_original_MFB" --logdir="../log/test_original_MFB"
python test_enet.py --checkpoint_dir="../log/train_original_ENet_combined_data" --logdir="../log/test_original_ENet_combined_data"
python test_enet.py --checkpoint_dir="../log/train_original_MFB_combined_data" --logdir="../log/test_original_MFB_combined_data"

# CamVid
python test_enet.py --checkpoint_dir="../log/train_original_ENet" --logdir="../log/test_original_ENet"
# Cityscapes
python test_enet.py --num_classes=20 --image_height=1024 --image_width=2048 --dataset="Cityscapes" --checkpoint_dir="../log/train_original_ENet_City" --logdir="../log/test_original_ENet_City"
#NYU
python test_enet.py --num_classes=5 --image_height=480 --image_width=640 --dataset="NYU" --checkpoint_dir="../log/train_original_ENet_NYU" --logdir="../log/test_original_ENet_NYU"
#ADE
python test_enet.py --num_classes=27 --image_height=480 --image_width=640 --dataset="ADE" --checkpoint_dir="../log/train_original_ENet_ADE" --logdir="../log/test_original_ENet_ADE"


# =============== DEMO ===============
# CamVid
python demo_enet.py --checkpoint_dir="../log/train_original_ENet" --logdir="../log/demo_original_ENet"
# Cityscapes
python demo_enet.py --num_classes=20 --image_height=1024 --image_width=2048 --dataset="000" --checkpoint_dir="../log/train_original_ENet_City" --logdir="../log/demo_original_ENet"
#NYU
python demo_enet.py --num_classes=5 --image_height=480 --image_width=640 --dataset="NYU" --checkpoint_dir="../log/train_original_ENet_NYU" --photo_dir="../log/demo_original_ENet_NYU"
#ADE
python demo_enet.py --num_classes=27 --image_height=480 --image_width=640 --dataset="ADE" --checkpoint_dir="../log/train_original_ENet_ADE" --logdir="../log/demo_original_ENet_ADE"

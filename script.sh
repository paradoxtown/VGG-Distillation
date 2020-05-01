# # train it
# echo --------train it-----------
# python train.py --lr 0.001 --ce True --it True --ht False --start_epoch 100 --load_student True --s_ckpt_path /home/jinze/vgg_distillation/checkpoint/distill/ckpt_1588003123_99_0.93134.pth

# # pretrain student model 90 epoch 0.01, 70 e 0.001
# echo --------pretrain student model-----------
# python pretrain.py --lr 0.01
# python pretrain.py --resume True --lr 0.001 --start_epoch 90 --s_ckpt_path /home/jinze/vgg_distillaton/ckpt_90_s.pth

# # train ht and it
# echo --------train ht and it-----------
# python train.py --lr 0.005 --load_student False --ce False --it False --ht True
# python train.py --lr 0.01 --load_student True --ce True --it True --ht False --start_epoch 100 --s_ckpt_path /home/jinze/vgg_distillation/distill/ckpt_100_ht.pth

# pretrain student
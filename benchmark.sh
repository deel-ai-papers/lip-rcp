
################################### CIFAR-10 #######################################

python scripts/train.py \
  --mixed_precision --wandb_log --temperature 5.0 --alpha 0.975 --min_margin 0.6 \
  --batch_size 512 --epochs 130 --p_dropout 0.0 --loss_fn shkr --on_gpu 1
python scripts/fast_rcp.py \
  --score_fn lac_sigmoid --epsilon 0.0 --alpha 0.1 --temp 5.0 --bias 0.0 \
  --dataset cifar10 --batch_size 500 --num_batches 10 --num_iters 10
python scripts/fast_rcp.py \
  --score_fn lac_sigmoid --epsilon 0.03 --alpha 0.1 --temp 0.1 --bias 0.0 \
  --dataset cifar10 --batch_size 500 --num_batches 10 --num_iters 10
python scripts/vcp_coverage.py
python scripts/poisoning.py

################################### ImageNet #######################################

# CUDA_VISIBLE_DEVICES=2,3 \
# torchrun --nproc_per_node=2 --master_port 25001 \
#   scripts/rcp_in1k.py \
#   --loss_fn shkr --temperature 10.0 --batch_size 1536 --optimizer adamw_sf \
#   --alpha 0.98 --min_margin 1.0 --wandb_log --img_size 224 \
#   --lr 2e-3 --weight_decay 1e-4 --epochs 150 --mixed_precision
# python scripts/fast_rcp.py \
#   --score_fn lac_sigmoid --epsilon 0.0 --alpha 0.1 --temp 5.0 --bias 0.0 \
#   --dataset imagenet --batch_size 500 --num_batches 30 --num_iters 5
# python scripts/fast_rcp.py \
#   --score_fn lac_sigmoid --epsilon 0.02 --alpha 0.1 --temp 5.0 --bias 0.0 \
#   --dataset imagenet --batch_size 500 --num_batches 30 --num_iters 5


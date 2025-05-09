source activate lranet
conda deactivate
source activate lranet
which python
CUDA_VISIBLE_DEVICES=1,2 setsid nohup bash  --gpus 2 > ./output/1.log 2>&1 &
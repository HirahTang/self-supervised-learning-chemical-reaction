#!/bin/bash
export PATH=/root/anaconda3/bin:$PATH
source activate mol3d
cd /share/project/tanghan/sharefs-hantang/tanghan/finetune_logs
export model_file=/share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/stage2_NodeEdge_no_stop_gradient_work_rc.pth
export model_type=NodeEdge_5_epoch
export mask_type=RC
export dataset_list=(tox21 hiv muv bace bbbp toxcast sider clintox);
# export dataset_list=(muv bace bbbp toxcast sider clintox);
export seed_list=(0 1 2);
for dataset in "${dataset_list[@]}"; do
    for seed in "${seed_list[@]}"; do
    export log_name=finetune_"$model_type"_"$mask_type"_"$dataset"_seed"$seed"
    python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/finetune/finetune_optuna.py --dataset "$dataset" --epochs 100 --n_trials 30 --input_model_file "$model_file" --runseed "$seed" --eval_train 1  --run_name "$model_type"_"$mask_type"_"$dataset" > /share/project/tanghan/sharefs-hantang/tanghan/finetune_logs/chemical_reaction_"$log_name".log;
    done;
done;
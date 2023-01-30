#!/bin/bash
export PATH=/root/anaconda3/bin:$PATH
source activate mol3d
cd /share/project/tanghan/sharefs-hantang/tanghan/finetune_logs
export model_file=/share/project/tanghan/sharefs-hantang/chem/model_gin/masking.pth
export model_type=orig
export mask_type=0
export dataset_list=(bbbp);
for dataset in "${dataset_list[@]}"; do
    export log_name=finetune_"$model_type"_"$mask_type"_"$dataset"
    python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/finetune/finetune_optuna.py --dataset "$dataset" --input_model_file "$model_file" --runseed 0 --eval_train 1  --run_name "$model_type"_"$mask_type"_"$dataset" > /share/project/tanghan/sharefs-hantang/tanghan/finetune_logs/chemical_reaction_finetune_"$log_name".log;
done;
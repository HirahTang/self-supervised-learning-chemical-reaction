#!/bin/bash
export PATH=/root/anaconda3/bin:$PATH
source activate mol3d
cd /share/project/tanghan/sharefs-hantang/tanghan/finetune_logs
export model_file=/share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/stage2_mlabe_no_stop_gradient_2three_hop.pth
export model_type=Mlabes
export mask_type=3hop
export dataset_list=(tox21 hiv muv bace bbbp toxcast sider clintox);
for dataset in "${dataset_list[@]}"; do
    export log_name=finetune_"$model_type"_"$mask_type"_"$dataset"
    python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/finetune/finetune_rcmasking.py --dataset "$dataset" --input_model_file "$model_file" --filename "$model_type"_"$mask_type"_no_stop_gradient --run_name "$model_type"_"$mask_type"_"$dataset" > /share/project/tanghan/sharefs-hantang/tanghan/finetune_logs/chemical_reaction_finetune_"$log_name".log;
done;
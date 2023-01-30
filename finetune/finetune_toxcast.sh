export PATH=/root/anaconda3/bin:$PATH
source activate mol3d
cd /share/project/tanghan/sharefs-hantang/tanghan/finetune_logs
export dataset_list=(toxcast);
for dataset in "${dataset_list[@]}"; do
    export log_name=finetune_NodeEdge_rc_"$dataset"
    python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/finetune/finetune_rcmasking.py --dataset "$dataset" --input_model_file /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/stage2_NodeEdge_no_stop_gradient_work_rc.pth --filename NodeEdge_rc_no_stop_gradient --run_name NodeEdge_rc_"$dataset" > /share/project/tanghan/sharefs-hantang/tanghan/finetune_logs/chemical_reaction_finetune_${log_name}.log
done
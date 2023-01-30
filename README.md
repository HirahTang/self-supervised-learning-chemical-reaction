# Chemical_Reaction_Pretraining

A self-supervised learning framework to exploit information from chemical reaction dataset



## attrmasking pretrain and testing

Stage one and testing code extracted from ```https://github.com/snap-stanford/pretrain-gnns```

Stage one(attrmasking strategy pretrain):

```
cd test
python test_gcn.py --output_model_file OUTPUT_MODEL_PATH

# adversarial pre-training, recommend value of adv_lr: 1e-2 ~ 1e-1
python test_gcn.py --adv --adv_lr ADV_LR --output_model_file OUTPUT_MODEL_PATH
```


Testing:
```
python finetune_gcn.py --input_model_file INPUT_MODEL_PATH --dataset DOWNSTREAM_DATASET{bbbp, sider...} --filename OUTPUT_FILE_PATH

```

## Stage 2 Continuous Self-supervised learning from Chemical Reaction Data

This module uses Hydra to organize configurations
Pre-train:
```
python pretrain/pretrain_rcmasking.py +experiment={Your experiment configuration(yaml)}
```
inside the YAML:
```
training_settings:
    device: The GPU index
    batch_size
    epochs_RC, epochs_1hop, epochs_2hop, epochs_3hop: Number of epochs for different phases
    lr, decay, seed
    stage2_on: if set 0, the model pre-trained from randomly initialized weights
    stage_one_model(path): if stage2_on is set 1, the model will be pre-trained from the model of this path
    validation_size(int): The size of validation set, the rest will be used as the training set.

model:
    num_layer, emb_dim, dropout_ratio, mask_rate, mask_edge, JK, gnn_type: GNN configurations
    output_model_file: The name of the trained models, it will in the format of {output_model_file}_{target}_{training_stage}.pth
    target: (mlabes, NodeEdge), the self-supervised task for pre-training.

wandb:
    login_key, run_name: You could set your own wandb settings here.
```
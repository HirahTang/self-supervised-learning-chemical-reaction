import torch
import sys
# sys.path.append(sys.path[0]+'/..')
sys.path.insert(0,'..')

import yaml

from models.graphormer import GraphormerEncoder
from datas.graphormer_data import MyZINC, GraphormerPYGDataset, BatchedDataDataset
from tqdm import tqdm



class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    # init model
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    config_file = 'assets/graphormer.yaml'
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
    model_config = Struct(**model_config)
    encoder = GraphormerEncoder(model_config).to(device)
    # init dataset 
    root = '/sharefs/sharefs-hantang/ZINC/dataset'
    inner_dataset = MyZINC(root=root)
    train_set = MyZINC(root=root, split="train")
    valid_set = MyZINC(root=root, split="val")
    test_set = MyZINC(root=root, split="test")
    seed = 0
    data_set = GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
    batched_data = BatchedDataDataset(
            data_set.train_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
        )

    # init dataloader
    data_loader = torch.utils.data.DataLoader(batched_data, batch_size=4, shuffle=True, num_workers = 1, collate_fn = batched_data.collater)
    # forward
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        for k in batch:
            batch[k] = batch[k].to(device)
        # batch = batch.to(device)
        node_rep = encoder(batch)
        print('finish')

    pass


if __name__ == "__main__":
    main()
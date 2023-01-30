#!/usr/bin/env python3

from pathlib import Path
import torch
import torchvision


def load_data(num_gpus):
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                 ])
    dataset = torchvision.datasets.ImageFolder(root=, transform=transforms)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4*num_gpus
                                                )
    return dataloader

def save_model(epoch, model, optimizer):
    """Saves model checkpoint on given epoch with given data name.
    """
    checkpoint_folder = Path.cwd() / 'model_checkpoints'
    if not checkpoint_folder.is_dir():
        checkpoint_folder.mkdir()
    file = checkpoint_folder / f'epoch_{epoch}.pt'
    if not file.is_file():
        file.touch()
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        file
                )
    return True

def load_model(epoch, model, optimizer):
    """Loads model state from file.
    """
    file = Path.cwd() / 'model_checkpoints' / f'epoch_{epoch}.pt'
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def run_training(num_gpus):

    model = torchvision.models.resnet50(pretrained=False)
    model = model.cuda()
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    model.train()
    num_epochs = 30
    dataloader = load_data(num_gpus)
    total_steps = len(dataloader)
    for epoch in range(1, num_epochs):
        print(f'\nEpoch {epoch}\n')
        if epoch > 1:
            model, optimizer = load_model(epoch-1, model, optimizer)
        for step, (images, labels) in enumerate(dataloader, 1):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f'Epoch [{epoch} / {num_epochs}], Step [{step} / {total_steps}], Loss: {loss.item():.4f}')
        save_model(epoch, model, optimizer)

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print('num_gpus: ', num_gpus)
    run_training(num_gpus)
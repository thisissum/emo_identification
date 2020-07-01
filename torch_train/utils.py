def seed_everything(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def move_to_device(data, device):
    output = []
    for item in data:
        if isinstance(item, (torch.Tensor, torch.cuda.Tensor)):
            output.append(item.to(device))
        elif isinstance(item, (tuple, list)):
            output += move_to_device(item, device)
        else:
            output.append(item)
    return output
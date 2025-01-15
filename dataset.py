from torch.utils.data import Dataset


class VGGdata(Dataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        data = self.data[index].float()
        target = self.targets[index].float()
        return data, target

    def __len__(self):
        return len(self.targets)

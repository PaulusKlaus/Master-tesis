import torch 
from torch.utils.data import Dataset, DataLoader


from .sequence_aug import *


class OneViewDataset(Dataset):
    def __init__(self, data_list, test = False, transform  =None):
        self.test = test

        if transform is None:
            self.transform = Compose([Reshape()])
        else:
            self.transform = transform

        self.seq_data = data_list['data'].tolist()
        if not self.test:
            self.labels = data_list['label'].tolist()


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        data = self.seq_data[idx]
        data = self.transform(data)

        if self.test:
            label = idx
        else:
            label = self.labels[idx]  # A bit unsure about this, can couse csome problems later 

        data = torch.tensor(data, dtype=torch.float32)  #torch.tensor is safer (no data leakage) then torch.as_tensor, but slower since it copier the data instead of using shared memory. 
        label = torch.tensor(label, dtype=torch.long)        
        return data, label


        
class TwoViewDataset(Dataset):
    def __init__(self, data_list, test = False, transform_1 = None, transform_2 =None):
        self.test = test

        if transform_1 is None:
            self.transform_1 = Compose([Reshape()])
        else:
            self.transform_1 = transform_1

        if transform_2 is None:
            self.transform_2 = Compose([Reshape()])
        else:
            self.transform_2 = transform_2

        self.seq_data = data_list['data'].tolist()
        if not self.test:
            self.labels = data_list['label'].tolist()

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        data = self.seq_data[idx]
        data_1 = self.transform_1(data)
        data_2 = self.transform_2(data)

        if self.test:
            label = idx
        else:
            label = self.labels[idx]  # A bit unsure about this, can couse csome problems later 

        data_1 = torch.tensor(data_1, dtype=torch.float32)  #torch.tensor is safer (no data leakage) then torch.as_tensor, but slower since it copier the data instead of using shared memory. 
        data_2 = torch.tensor(data_2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)        
        return data_1, data_2, label



class MultiViewDataset(Dataset):
    def __init__(self, data_list, transforms, test=False, return_index=False):
        """
        transforms: list/tuple of callables, e.g. [t1] or [t1, t2]
        test: if True, no label expected
        return_index: optionally return idx as well (useful for evaluation/reordering)
        """
        self.test = test
        self.return_index = return_index

        if not isinstance(transforms, (list, tuple)) or len(transforms) == 0:
            self.transforms = (Compose([Reshape()]))
            #raise ValueError("transforms must be a non-empty list/tuple of callables")

        self.transforms = transforms
        self.seq_data = data_list["data"].tolist()

        if not self.test:
            self.labels = data_list["label"].tolist()

    def __len__(self):
        return len(self.seq_data)

    def _to_float_tensor(self, x):
        # Safe + consistent dtype. Copying is usually OK here.
        return torch.tensor(x, dtype=torch.float32)

    def __getitem__(self, idx):
        x = self.seq_data[idx]

        views = [self._to_float_tensor(t(x)) for t in self.transforms]

        if self.test:
            y = idx  # or None, depending on your preference
        else:
            y = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.return_index:
            return (*views, y, idx)
        return (*views, y)
from .base import Places205Dataset
from hirl.runners import FinetuneRunner

class Places205FinetuneRunner(FinetuneRunner):
    def build_train_dataset(self, args, transform):
        dataset = Places205Dataset(args.data_path, "train", transform)
        return dataset

    def build_val_dataset(self, args, transform):
        dataset = Places205Dataset(args.data_path, "val", transform)
        return dataset
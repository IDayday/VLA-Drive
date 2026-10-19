from copy import deepcopy
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Subset
import numpy as np

def mix_datasets(datasets, names, ratios, total=None):
    if total is None:
        total = min(int(len(ds) / ratios[n]) for ds, n in zip(datasets, names))

    return ConcatDataset([
        Subset(ds, np.random.choice(len(ds), int(ratios[n] * total), replace=False))
        for ds, n in zip(datasets, names)
    ])


class GeneralDataModule(pl.LightningDataModule):
    default_train_loader_opts = DictConfig(
        {
            "batch_size": 4,
            "num_workers": 4,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            # "persistent_workers": True,
        }
    )
    default_val_loader_opts = DictConfig(
        {
            "batch_size": 1,
            "num_workers": 4,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False,
            # "persistent_workers": True,
        }
    )

    def __init__(
        self,
        train_dataset: DictConfig = None,
        val_dataset: DictConfig = None,
        test_dataset: DictConfig = None,
        train_loader_opts: DictConfig = None,
        val_loader_opts: DictConfig = None,
        **kwargs,
    ):
        """
        Initialize the GeneralDataModule with datasets and loader options.

        This is a general datamodule that can be used for any dataset.
        Train uses ConcatDataset. Val and Test use CombinedLoader, sequentially
        consuming each iterable and returning a triplet (data, idx, iterable_idx).

        Args:
            train_dataset (DictConfig): Configuration for the training dataset.
            val_dataset (DictConfig): Configuration for the validation dataset.
            train_loader_opts (DictConfig): Options for the training data loader.
            val_loader_opts (DictConfig): Options for the validation data loader.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader_opts = self.default_train_loader_opts
        self.val_loader_opts = self.default_val_loader_opts

        if train_loader_opts is not None:
            self.train_loader_opts.update(train_loader_opts)
        if val_loader_opts is not None:
            self.val_loader_opts.update(val_loader_opts)

    def val_dataloader(self):
        """
        Create and return the validation data loader.

        Returns:
            CombinedLoader or DataLoader: The validation data loader.
        """
        loaders = GeneralDataModule._parse_loaders(self.val_dataset, self.val_loader_opts)
        if isinstance(loaders, list):
            return CombinedLoader(loaders, mode="sequential")
        else:
            return loaders

    def train_dataloader(self):
        """
        Create and return the training data loader.

        Returns:
            DataLoader: The training data loader.
        """
        return GeneralDataModule._parse_train_dataloader(self.train_dataset, self.train_loader_opts)

    @staticmethod
    def _parse_train_dataloader(config, loader_opts):
        """
        Parse and create the training data loader from the configuration.

        Args:
            config (DictConfig): Configuration for the dataset.
            loader_opts (DictConfig): Options for the data loader.

        Returns:
            DataLoader or CombinedLoader: The training data loader.
        """
        if isinstance(config.dataset_opts, ListConfig):
            datasets = GeneralDataModule._parse_datasets(config)
            if config.pretrain:
                names = ["hypersim"]
                ratios = {"hypersim": 1.0}
            else:
                names = ["hypersim", "urbansyn", "unrealstereo4k", "vkitti", "tartanair"]
                ratios = {"hypersim": 0.5, "urbansyn": 0.15, "unrealstereo4k": 0.15, "vkitti": 0.1, "tartanair": 0.1}
            dataset = mix_datasets(datasets, names, ratios, total=48000)
            return DataLoader(dataset, **loader_opts)
        else:
            return GeneralDataModule._parse_loaders(config, loader_opts)

    @staticmethod
    def _parse_datasets(config):
        """
        Parse and instantiate datasets from the configuration.

        Args:
            config (DictConfig): Configuration for the datasets.

        Returns:
            list: A list of instantiated datasets.
        """
        datasets = []
        for idx, dataset_opt in enumerate(config.dataset_opts):
            dataset = instantiate(dataset_opt)
            datasets.append(dataset)
        return datasets

    @staticmethod
    def _parse_loaders(config, loader_opts):
        """
        Parse and create data loaders from the configuration.

        Args:
            config (DictConfig): Configuration for the datasets.
            loader_opts (DictConfig): Options for the data loaders.

        Returns:
            DataLoader or list: A single DataLoader or a list of DataLoaders.
        """
        if not isinstance(config.dataset_opts, ListConfig):
            dataset = instantiate(config.dataset_opts)
            if "loader_opts" in config:
                loader_opts = deepcopy(loader_opts)
                loader_opts.update(config.loader_opts)
            return DataLoader(dataset, **loader_opts)
        else:
            dataloaders = []
            for idx, dataset_opt in enumerate(config.dataset_opts):
                if isinstance(dataset_opt, ListConfig):
                    datasets = [instantiate(opt) for opt in dataset_opt]
                    dataset = ConcatDataset(datasets)
                else:
                    dataset = instantiate(dataset_opt)
                if "loader_opts" in config:
                    loader_opt = deepcopy(loader_opts)
                    if isinstance(config.loader_opts, ListConfig):
                        loader_opt.update(config.loader_opts[idx])
                    else:
                        loader_opt.update(config.loader_opts)
                dataloaders.append(DataLoader(dataset, **loader_opts))
            return dataloaders
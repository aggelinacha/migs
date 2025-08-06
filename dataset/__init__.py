from .zjumocap import ZJUMoCapDataset, ZJUMoCapDatasetMulti

def load_dataset(cfg, split='train'):
    dataset_dict = {
        'zjumocap': ZJUMoCapDataset,
        'zjumocap_multi': ZJUMoCapDatasetMulti,
    }
    return dataset_dict[cfg.name](cfg, split)

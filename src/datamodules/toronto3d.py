# src/datamodules/toronto3d.py
import logging
from src.datamodules.base import BaseDataModule
from src.datasets import Toronto3D, MiniToronto3D  # 🟡 修改导入的dataset类

log = logging.getLogger(__name__)

class Toronto3DDataModule(BaseDataModule):  # 🟡 修改类名
    """LightningDataModule for Toronto-3D dataset."""
    
    _DATASET_CLASS = Toronto3D          # 🟡 指定主数据集类
    _MINIDATASET_CLASS = MiniToronto3D  # 🟡 指定迷你数据集类

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/datamodule/semantic/toronto3d.yaml")  # 🟡 修改配置文件路径
    cfg.data_dir = root + "/data"
    _ = hydra.utils.instantiate(cfg)
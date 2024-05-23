import h5py as h5
import numpy as np
import logging
import yaml
import typing as tp


class NopartsHandler:
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file
        self.file = None
        self.out_file = None
        self.name = self.collector.get_h5_name()
        print(f"{self.name=}")
    
    @staticmethod
    def _load_cfg(path='./settings/cfg_unzip.yml'):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cfg
    
    @classmethod
    def from_cfg(cls, path_to_cfg):
        cfg = cls._load_cfg(path_to_cfg)
        return cls(cfg['path_to_file'])
    
    def add_pk(self):
        self.file = h5.File(self.path_to_raw, 'a')
        starts = self.file['ev_starts'][:]
        hits_len = starts[-1]
        assert hits_len == self.file['data'].shape[0], f"{hits_len=}, data_shape={self.file['data'].shape[0]}"
        pk = np.arange(len(starts)-1, dtype=np.int64)
        
        pk_hits = np.zeros(hits_len, dtype=np.int64)
        for i, (s,e) in enumerate(zip(starts[0:-1], starts[1:])):
            assert e>s, "Wrong ev starts sequence"
            pk_hits[s:e] = pk[i]
        assert pk_hits[-1] > 0, "Wrong pk_hits enumeration"
        self.file.create_dataset('pk', data=pk, shape=pk.shape, dtype=pk.dtype)
        self.file.create_dataset('pk_hits', data=pk_hits, shape=pk_hits.shape, dtype=pk_hits.dtype)
        self.file.close()
        
    def add_prty_labels(self):
        self.file = h5.File(self.path_to_raw, 'a')
        ids = self.file['ev_ids'][:]
        p_labels = np.zeros(len(ids), dtype=np.int8)
        mask_nuatm = np.char.startswith(ids, b'nuatm')
        mask_nu2 = np.char.startswith(ids, b'nu2')
        p_labels[mask_nuatm] = 1
        p_labels[mask_nu2] = 2
        self.file.create_dataset('p_labels', data=p_labels, shape=p_labels.shape, dtype=p_labels.dtype)
        self.file.close()
    
import h5py as h5
import numpy as np
import yaml
import logging
import typing as tp
from datetime import datetime

class RawHandler:
    def __init__(self, path_to_raw: str, path_to_out: str, keys_to_copy: list[str], keys_with_parts: list[str], bad_parts: dict[list]):
        self.path_to_raw = path_to_raw
        self.path_to_out = path_to_out
        self.file = None
        self.out_file = None
        self.keys_to_copy=keys_to_copy
        self.keys_with_parts = keys_with_parts
        self.bad_parts = bad_parts
        
    @staticmethod
    def _load_cfg(path='./settings/pk_cfg.yml'):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cfg
    
    @classmethod
    def from_cfg(cls, path_to_cfg):
        cfg = cls._load_cfg(path_to_cfg)
        return cls(cfg['path_to_raw'], cfg['path_to_out'], cfg['keys_to_copy'], cfg['keys_with_parts'], cfg['bad_parts'])
    
    def _iterate_over_parts(self, data_list:list, elp: str, key_self: str, bad_parts: list):
        # iterate over parts in file, collect data
        logging.info(f'\t\t{key_self=}')
        for part in self.file[f"{elp}/{key_self}"].keys():
            if part[5:9] not in bad_parts:
                logging.debug(f"{part=}")
                if key_self.startswith('muons_prty'):
                    data_list.append(self.file[f'{elp}/{key_self}/{part}/data'][:, -1])
                elif key_self == "raw/ev_starts":
                    data_list.append(np.array(self.file[f'{elp}/{key_self}/{part}/data'][:], dtype=np.int64))
                else:
                    data_list.append(self.file[f'{elp}/{key_self}/{part}/data'][:])
        return data_list
    
    @staticmethod
    def _concat_datalist(data_list: list, key: str) -> np.ndarray:
        if key == 'raw/ev_starts':
            ev_starts_list = [data_list[0][:] - data_list[0][0]]
            for i, d in enumerate(data_list[1:]):
                assert sum(np.diff(d)<=0)==0, logging.warning(f"Wrongs ev starts at {i=}:\n{d}")
                ev_starts_in_part = ev_starts_list[i][-1:] + (d[1:]-d[0:1])
                ev_starts_list.append(ev_starts_in_part)
                assert ev_starts_list[i][-1]<ev_starts_list[i+1][0], f"{i+1=},\n {ev_starts_in_part[0:10]=}\n, {ev_starts_list[i][-1:].dtype=},\n {ev_starts_list[i][-10:]=},\n {ev_starts_list[i+1][0:10]=}\n{d[0:10]=}"
            data = np.concatenate(ev_starts_list, axis=0)
        else:
            data = np.concatenate(data_list, axis=0)
        return data
    
    @staticmethod
    def _transform_key_to_out(key):
        if key.startswith("raw/"):
                key = key[4:] # remove raw/ prefix
        if key == 'labels':
            key = 'is_signal'
        return key
        
    def unzip_parts(self) -> None:
        logging.basicConfig(filename='./logs/UnParting.log', filemode='w', level=logging.INFO)
        t0 = datetime.now()
        logging.info(f'{t0=}')
        self.file = h5.File(self.path_to_raw, 'r')
        self.out_file = h5.File(self.path_to_out, 'w')
        keys_elpart = self.file.keys()
        for key in self.keys_to_copy:
            logging.info(f"{key=}")
            for elp in keys_elpart:
                logging.info(f"\t{elp=}")
                data = self.file[f'{elp}/{key}/data']
                self.out_file.create_dataset(f'{elp}/{key}', data=data, shape=data.shape, dtype=data.dtype)
        for key in self.keys_with_parts:
            data_list = [] # dont move it. We want to collect data for all elp in the list
            logging.info(f"{key=}")
            for elp in keys_elpart:
                logging.info(f"\t{elp=}")
                # muatm events have many individ muons. Will replace with aggregate
                if key == 'muons_prty/individ' and elp=='muatm':
                    key_self = 'muons_prty/aggregate' # key of self.file
                else:
                    key_self = key
                data_list = self._iterate_over_parts(data_list, elp=elp, key_self=key_self, bad_parts=self.bad_parts[elp])
            # ev_starts processed separately
            data = self._concat_datalist(data_list, key=key)
            # load data to new file
            key = self._transform_key_to_out(key)
            self.out_file.create_dataset(f'{key}', data=data, shape=data.shape, dtype=data.dtype)
        print("H5 file unzipped!")
        logging.info(f'Passed time: {datetime.now()-t0}')
        self.file.close()
        self.out_file.close()

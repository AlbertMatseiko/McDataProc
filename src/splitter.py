import h5py as h5
import numpy as np
import yaml
import logging
from datetime import datetime


class SplitterNS:
    def __init__(self, path_to_h5: str, path_to_out: str, 
                 num_train_mu: int, num_test_mu: int, num_val_mu: int,
                 num_train_nu: int, num_test_nu: int, num_val_nu: int,
                 hits_keys: list[str], events_keys: list[str]):
        # files
        self.path_to_h5 = path_to_h5
        self.path_to_out = path_to_out
        self.file = None
        self.out_file = None
        # muons
        self.num_train_mu = num_train_mu
        self.num_test_mu = num_test_mu
        self.num_val_mu = num_val_mu
        # neutrinos
        self.num_train_nu = num_train_nu
        self.num_test_nu = num_test_nu
        self.num_val_nu = num_val_nu
        # keys
        self.hits_keys = hits_keys
        self.events_keys = events_keys
    
    @staticmethod
    def _load_cfg(path='./settings/cfg_splitNS.yml'):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cfg

    @classmethod
    def from_cfg(cls, path_to_cfg):
        cfg = cls._load_cfg(path_to_cfg)
        return cls(**cfg)
    
    @staticmethod
    def _shuffle_data(permutation: np.ndarray[int], data: np.ndarray[np.float32], ev_starts: np.ndarray[int]):
        starts = ev_starts[:-1]
        ends = ev_starts[1:]
        data_new = np.zeros(data.shape, dtype=np.float32)
        current_data_new_index = 0
        for idx in permutation:
            start, stop = starts[idx], ends[idx]
            length = stop - start
            assert length>0, f"{idx=}, {start=}, {stop=}"
            data_new[current_data_new_index:current_data_new_index + length] = data[start:stop]
            current_data_new_index += length
        logging.info(f"\t\tdata sample shuffled")
        return data_new

    @staticmethod
    def _idxs_to_mask(idxs: np.ndarray[int], length: int) -> np.ndarray[bool]:
        mask = np.zeros(length, dtype=bool)
        mask[idxs] = True
        logging.info(f"\t\tidxs tranfromed into mask")
        return mask
    
    @staticmethod
    def _get_new_starts(starts: np.ndarray[int], idxs: np.ndarray[int]) -> np.ndarray[int]:
        lens = np.diff(starts)
        new_starts = np.zeros(len(idxs)+1, dtype=np.int64)
        for i, l in enumerate(lens[idxs]):
            assert l>0, f"{i=}, {l=}"
            new_starts[i+1] = new_starts[i] + l
        logging.info(f"\t\tstarts recalculated according to new indexes")
        return new_starts
    
    def _idxs_from_ids(self, ev_ids):
        idxs_mu = np.where(np.char.startswith(ev_ids, b'mu'))[0]
        idxs_mu_train = idxs_mu[:self.num_train_mu]
        idxs_mu_test = idxs_mu[self.num_train_mu:self.num_train_mu+self.num_test_mu]
        if self.num_val_mu is None:
            idxs_mu_val = idxs_mu[self.num_train_mu+self.num_test_mu:]
        else:
            idxs_mu_val = idxs_mu[self.num_train_mu+self.num_test_mu:self.num_train_mu+self.num_test_mu+self.num_val_mu]
        
        idxs_nu = np.where(np.char.startswith(ev_ids, b'nu'))[0]
        idxs_nu_train = idxs_nu[:self.num_train_nu]
        idxs_nu_test = idxs_nu[self.num_train_nu:self.num_train_nu+self.num_test_nu]
        if self.num_val_nu is None:
            idxs_nu_val = idxs_nu[self.num_train_nu+self.num_test_nu:]
        else:
            idxs_nu_val = idxs_nu[self.num_train_nu+self.num_test_nu:self.num_train_nu+self.num_test_nu+self.num_val_nu]
        
        idxs_train = np.concatenate([idxs_mu_train, idxs_nu_train], axis=0)
        idxs_test = np.concatenate([idxs_mu_test, idxs_nu_test], axis=0)
        idxs_val = np.concatenate([idxs_mu_val, idxs_nu_val], axis=0)
        logging.info("idxs train-test-val collected")
        return idxs_train, idxs_test, idxs_val
    
    def split_h5(self):
        logging.basicConfig(filename=f'./logs/{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}_splitting.log', filemode='w', level=logging.INFO)
        t0 = datetime.now()
        logging.info(f'{t0=}')
        self.file = h5.File(self.path_to_h5, 'r')
        self.out_file = h5.File(self.path_to_out, 'w')
        ev_ids = self.file['ev_ids'][:]
        logging.info("ev_ids collected")
        idxs_train, idxs_test, idxs_val = self._idxs_from_ids(ev_ids)
        perm_train = np.random.permutation(idxs_train.shape[0])
        
        for ev_key in self.events_keys:
            logging.info(f"{ev_key=}")
            sample = self.file[ev_key][:]
            logging.info(f"\tgot sample")
            if ev_key == 'ev_starts':
                self.out_file.create_dataset(f"train/ev_starts", data=self._get_new_starts(sample, idxs_train[perm_train]))
                logging.info(f"\ttrain ev_starts written")
                self.out_file.create_dataset(f"test/ev_starts", data=self._get_new_starts(sample, idxs_test))
                logging.info(f"\ttest ev_starts written")
                self.out_file.create_dataset(f"val/ev_starts", data=self._get_new_starts(sample, idxs_val))
                logging.info(f"\tval ev_starts written")
            else:
                self.out_file.create_dataset(f"train/{ev_key}", data=sample[idxs_train[perm_train]])
                logging.info(f"\ttrain {ev_key=} written")
                self.out_file.create_dataset(f"test/{ev_key}", data=sample[idxs_test])
                logging.info(f"\ttest {ev_key=} written")
                self.out_file.create_dataset(f"val/{ev_key}", data=sample[idxs_val])
                logging.info(f"\tval {ev_key=} written")
        
        num_ev = len(ev_ids)
        ev_starts = self.file['ev_starts'][:]
        lengths = np.diff(ev_starts)
        mask_hits_train = np.repeat(self._idxs_to_mask(idxs_train, num_ev), lengths)
        mask_hits_test = np.repeat(self._idxs_to_mask(idxs_test, num_ev), lengths)
        mask_hits_val = np.repeat(self._idxs_to_mask(idxs_val, num_ev), lengths)
        logging.info("hits masks train-test-val collected")
        for h_key in self.hits_keys:
            logging.info(f"{h_key=}")
            sample = self.file[h_key][:]
            logging.info(f"\tgot sample")
            self.out_file.create_dataset(f"train/{h_key}", data=self._shuffle_data(perm_train, sample[mask_hits_train], self._get_new_starts(ev_starts, idxs_train))) 
            logging.info(f"\ttrain {h_key=} written")
            self.out_file.create_dataset(f"test/{h_key}", data=sample[mask_hits_test])
            logging.info(f"\ttest {h_key=} written")
            self.out_file.create_dataset(f"val/{h_key}", data=sample[mask_hits_val])  
            logging.info(f"\tval {h_key=} written")
        logging.info(f'Passed time: {datetime.now()-t0}')
        self.file.close()
        self.out_file.close()
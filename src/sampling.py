import yaml
import numpy as np
import typing as tp
import h5py as h5
import yaml
import logging
from datetime import datetime


class Sampler:
    def __init__(self, path_to_h5: str, path_to_out: str, filters: dict, hits_keys: list, events_keys: list):
        self.path_to_h5 = path_to_h5
        self.path_to_out = path_to_out
        self.filters = filters
        self.hits_keys = hits_keys
        self.events_keys = events_keys
        self.file = None
        self.out_file = None
    
    @staticmethod
    def _load_cfg(path: str = './settings/filter_cfg.yml') -> dict:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cfg
    
    @classmethod
    def from_cfg(cls, path_to_cfg: str):
        cfg = cls._load_cfg(path_to_cfg)
        return cls(cfg['path_to_h5'], cfg['path_to_out'], 
                   filters=cfg['filters'], 
                   hits_keys=cfg['hits_keys'], 
                   events_keys=cfg['events_keys'])
    
    @staticmethod
    def _count_lens(starts: np.ndarray[int], mask_hits: tp.Optional[np.ndarray[bool]] = None) -> np.ndarray[int]:
        logging.info(f"Start _count_lens")
        if mask_hits is None:
            mask_hits = np.ones(len(starts), dtype=bool)
        lengths_list = [np.sum(mask_hits[s:e]) for s,e in zip(starts[:-1],starts[1:])]
        logging.info(f"\t{len(lengths_list)=}")
        return np.array(lengths_list, dtype=np.int32)
    
    @staticmethod
    def _count_un_strings(starts: np.ndarray[int], tr_chs: np.ndarray[int], mask_hits: tp.Optional[np.ndarray[bool]] = None) -> np.ndarray[int]:
        logging.info(f"Start _count_un_strings")
        if mask_hits is None:
            mask_hits = np.ones(len(starts), dtype=bool)
        tr_chs_masked = np.where(mask_hits, tr_chs, -1)
        tr_chs_masked = tr_chs_masked//36
        logging.info(f"\t{tr_chs_masked.shape=}")
        nums_un_strings = [(np.unique(tr_chs_masked[s:e])>0).sum() for s,e in zip(starts[:-1], starts[1:])]
        logging.info(f"\t{len(nums_un_strings)=}")
        return np.array(nums_un_strings, dtype=np.int32)

    @staticmethod
    def _idxs_flatten_spec(lgE: np.ndarray, start=None, stop=None, bins=None) -> np.ndarray[int]:
        '''
        idxs of events to make flat energy spectra
        '''
        if start is None:
            start = 1
        if stop is None:
            stop = 6
        if bins is None:
            bins = 20
        e_range = np.linspace(start, stop, bins+1)
        e_range = np.concatenate([[-10.], e_range, [10.]], axis=0)
        nums = []
        for bin_start, bin_stop in zip(e_range[:-1], e_range[1:]):
            nums.append(lgE[(lgE>bin_start) * (lgE<=bin_stop)].shape[0])
        num_per_bin = min(nums)
        idxs = np.array([], dtype=int)
        for bin_start, bin_stop in zip(e_range[:-1], e_range[1:]):
            bin_mask = ((lgE>bin_start) * (lgE<=bin_stop)) #masks all events inside current bin
            idxs_bin = np.where(bin_mask)[0]
            idxs = np.append(idxs, idxs_bin[:num_per_bin], axis=0) #append (only a number) of global idxs of events inside current bin
        return np.random.permutation(idxs)
        
    def _get_masks_for_file(self) -> tuple[np.ndarray[bool], np.ndarray[bool]]:
        starts = self.file['ev_starts'][:]
        mask_hits = np.ones(self.file['data'].shape[0], dtype=bool)
        if self.filters['Q'][0]>0 or self.filters['Q'][1]<10**9:
            Q = self.file['data'][:,0]
            mask_hits *= (Q>=self.filters['Q'][0])*(Q<self.filters['Q'][1])
            logging.info("Q filter applied")
        if self.filters['only_signal'] == True:
            sig_labels = self.file['is_signal'][:]
            mask_hits *= (sig_labels!=0)
            logging.info("Signal filter applied")
        mask_events = np.ones(self.file['ev_ids'].shape[0], dtype=bool)
        if self.filters['only_nu'] == True:
            ev_ids = self.file['ev_ids'][:]
            mask_events *= np.char.startswith(ev_ids, b'nu')
            assert mask_events.sum()>0
            logging.info("Particle=neutrino filter applied")
        if self.filters['E_prime'][0]>0 or self.filters['E_prime'][1]<10**9:
            E_prime = self.file['prime_prty'][:,2]
            mask_events *= (E_prime>=self.filters['E_prime'][0])*(E_prime<self.filters['E_prime'][1])
            logging.info("E_prime filter applied")
        if self.filters['E_mu'][0]>0 or self.filters['E_mu'][1]<10**9:
            E_mu = self.file['muons_prty/individ'][:]
            mask_events *= (E_mu>=self.filters['E_mu'][0])*(E_mu<self.filters['E_mu'][1])
            logging.info("E_prime filter applied")
        if self.filters['hits'][0]>0 or self.filters['hits'][1]<10**9:
            lens = self._count_lens(starts, mask_hits)
            mask_events *= (lens>=self.filters['hits'][0])*(lens<self.filters['hits'][1])
            logging.info("hits filter applied")
        if self.filters['strings'][0]>0 or self.filters['strings'][1]<10**9:
            tr_chs = self.file['channels'][:]
            un_str = self._count_un_strings(starts, tr_chs, mask_hits)
            mask_events *= (un_str>=self.filters['strings'][0])*(un_str<self.filters['strings'][1])
            logging.info("strings filter applied")
        if self.filters['make_flat_E']==True:
            E = self.file['muons_prty/individ'][:][mask_events]
            E[E<=0] = 1e-3
            lgE = np.log10(E)
            idxs_to_flat = self._idxs_flatten_spec(lgE, bins=20)
            assert len(idxs_to_flat)>0
            idxs_events = np.where(mask_events)[0][idxs_to_flat]
            assert len(idxs_events)>0
            mask_events = np.zeros(mask_events.shape[0], dtype=bool)
            mask_events[idxs_events] = True
            assert mask_events.sum()>0
            logging.info("flat spectrum filter applied")
        mask_hits *= np.repeat(mask_events, np.diff(starts)) #self._update_mask_hits(mask_hits, mask_events, starts)
        logging.info("mask_hits updated")
        return mask_hits, mask_events

    def _get_new_starts(self, starts: np.ndarray[int], mask_hits: np.ndarray[bool]) -> np.ndarray[int]:
        lens = self._count_lens(starts, mask_hits)
        new_starts = [0]
        for l in lens:
            if l>0:
                new_starts.append(new_starts[-1] + l)
        return np.array(new_starts, dtype=np.int64)
    
    def cretate_h5(self) -> None:
        """
        Creates a filtered HDF5 file according to initialized settings.
        The created file will be stored at 'h5_files' folder.
        """
        logging.basicConfig(filename='./logs/Filtering.log', filemode='w', level=logging.INFO)
        t0 = datetime.now()
        logging.info(f'{t0=}')
        self.file = h5.File(self.path_to_h5, 'r')
        self.out_file = h5.File(self.path_to_out, 'w')
        mask_hits, mask_events = self._get_masks_for_file()
        logging.info(f"Got masks for file: {mask_events.shape=}, {mask_hits.shape=}")
        for h_key in self.hits_keys:
            sample = self.file[h_key][:]
            sample = sample[mask_hits]
            logging.info(f"Got sample for {h_key=}. {sample.shape=}")
            self.out_file.create_dataset(h_key, data=sample, shape=sample.shape, dtype=sample.dtype)
            logging.info(f"Dataset for {h_key=} is created")
        for ev_key in self.events_keys:
            sample = self.file[ev_key][:]
            if ev_key == "ev_starts":
                sample = self._get_new_starts(sample, mask_hits)
                assert sample.shape[0] - 1 == self.out_file['ev_ids'].shape[0]
                assert sum(np.diff(sample)<0)==0
                assert sample[-1] == self.out_file['data'].shape[0]
            else: 
                sample = sample[mask_events]
            logging.info(f"Got sample for {ev_key=}. {sample.shape=}")
            self.out_file.create_dataset(ev_key, data=sample, shape=sample.shape, dtype=sample.dtype)
            logging.info(f"Dataset for {ev_key=} is created.")
        logging.info(f'Passed time: {datetime.now()-t0}')
        self.file.close()
        self.out_file.close()
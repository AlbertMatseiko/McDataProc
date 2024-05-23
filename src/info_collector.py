import os
import sys
from io import StringIO
from show_h5 import show_h5
import h5py as h5
import yaml
import numpy as np

class H5Info:
    
    def __init__(self, path_to_h5: str = "../h5_files/baikal_multi_1223_flat.h5"):
        self.path = path_to_h5

    def _get_structure_h5(self):
        stdout = sys.stdout
        sys.stdout = StringIO()
        show_h5(self.path, show_attrs=True, show_data='none')
        x = sys.stdout.getvalue()
        sys.stdout = stdout
        return x

    def get_h5_name(self):
        for i, c in enumerate(self.path[::-1]):
            if c == '/':
                break
        h5_name = self.path[-i:-3]
        return h5_name

    def get_prty_nums(self, path_to_ids: str = 'ev_ids') -> tuple[int, int, int]:
        with h5.File(self.path, 'r') as hf:
            try:
                ev_ids = hf[path_to_ids][:]
                num_muatm = (np.char.startswith(ev_ids, b'mu')).sum()
                num_nuatm = (np.char.startswith(ev_ids, b'nuatm')).sum()
                num_nu2 = (np.char.startswith(ev_ids, b'nu2')).sum()
                return num_muatm, num_nuatm, num_nu2
            except Exception as e:
                print(f"Doesn't work with this file. Try on file with {path_to_ids=}. {e=}")
                return None, None, None
    
    def add_norm_params(self):
        try:
            file = h5.File(self.path, 'a')
            data = file['train/data'][:]
            data = np.array(data, dtype=np.float64)
            mean, std = data.mean(axis=0), data.std(axis=0)
            try:
                del file['norm_params']
            except:
                pass
            file.create_dataset('norm_params/mean', data=mean, dtype=np.float32)
            file.create_dataset('norm_params/std', data=std, dtype=np.float32)
            file.close()
        except Exception as e:
           print(f"Doesn't work with this file. Maybe train-test-val splitting wasn't provided.\n{e=}")

    def collect(self, path_to_out: str = "../h5_files/baikal_multi_1223_flat.txt", mode: str = 'w'):
        size = os.stat(self.path)[6]
        print(f"Size is {size/2**30:.2f} GB", file=open(path_to_out, mode))
        
        self.file = h5.File(self.path, 'r')
        
        keys1 = list(self.file.keys())
        print(f"Keys1: {keys1}", file=open(path_to_out,'a'))
        try:
            keys2 = list(self.file[keys1[0]].keys())
            print(f"Keys2: {keys2}", file=open(path_to_out,'a'))
        except:
            pass
        self.file.close()
        x = self._get_structure_h5()
        print(x, file=open(path_to_out,'a'))
    
    @staticmethod
    def save_cfg(path_to_cgf: str, path_to_out: str) -> None:
        with open(path_to_cgf, 'r') as f:
            cfg = yaml.safe_load(f)
        with open(path_to_out, 'w') as f:
            yaml.safe_dump(cfg, f)
            

            

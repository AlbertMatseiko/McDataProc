from src.splitter import SplitterNS
from src.info_collector import H5Info

# filter data and save in distinct hdf5
path_to_cfg = "./settings/split_cfg.yml"
s = SplitterNS.from_cfg(path_to_cfg)
s.split_h5()
ic = H5Info(s.path_to_out)
ic.add_norm_params()
for regime in ['train','test','val']:
    muatm, nuatm, nu2 = ic.get_prty_nums(f"{regime}/ev_ids")
    if regime == 'train': mode = 'w' 
    else: mode = 'a'
    print(f"{regime=}:\n{muatm=}, {nuatm=}, {nu2=}", file=open(f"{s.path_to_out[:-3]}.txt", mode))
ic.collect(f"{s.path_to_out[:-3]}.txt", mode='a')
ic.save_cfg(path_to_cfg, f"{s.path_to_out[:-3]}.cfg")


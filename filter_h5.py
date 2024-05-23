from src.sampling import Sampler
from src.info_collector import H5Info

# filter data and save in distinct hdf5
path_to_cfg = "./settings/filter_cfg.yml"
s = Sampler.from_cfg(path_to_cfg)
s.cretate_h5()

# collect and save some information about the created file 
ic = H5Info(s.path_to_out)
muatm, nuatm, nu2 = ic.get_prty_nums(f"ev_ids")
print(f"{muatm=}, {nuatm=}, {nu2=}\n", file=open(f"{s.path_to_out[:-3]}.txt", 'w'))
ic.collect(f"{s.path_to_out[:-3]}.txt", mode='a')
ic.save_cfg(path_to_cfg, f"{s.path_to_out[:-3]}.cfg")
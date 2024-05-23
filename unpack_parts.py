from src.raw_handler import RawHandler
from src.info_collector import H5Info

# raw flat h5 file transform
path_to_cfg = "./settings/unzip_cfg.yml"
rh = RawHandler.from_cfg(path_to_cfg)
rh.unzip_parts() # creates new file
ic = H5Info(rh.path_to_raw)
ic.collect(f"{rh.path_to_raw[:-3]}.txt")
ic = H5Info(rh.path_to_out)
ic.collect(f"{rh.path_to_out[:-3]}.txt")
ic.save_cfg(path_to_cfg, f"{rh.path_to_out[:-3]}.cfg")
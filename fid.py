import pyiqa
import torch



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




# For FID metric, use directory or precomputed statistics as inputs
# refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
fid_metric = pyiqa.create_metric('fid')
score = fid_metric("/NADB/results/edges2handbags/samples_nfe119/images/","/edges2handbags/x0_64/")
print(score)
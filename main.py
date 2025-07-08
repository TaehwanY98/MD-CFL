from Network.GRU import GRU, K_emo_GRU
from server.FedAvgServer import FedAvgServer
from utils import Wesad_Train as wesad
from utils import Kemo_Train as kemo
from utils.parser import Simulationparser
from utils.CustomDataset import WESADDataset, KemoDataset
import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import os
from flwr.common import Context
from client.client import FedAvgClient
from server.ClusteredFedAvgServer import ClusteredFedAvg
import warnings
import pandas as pd
def set_seeds(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def make_model_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Simulationparser()
set_seeds(args)
Lossf = torch.nn.BCEWithLogitsLoss().to(DEVICE)
CosVsMaha = {"cosine":[], "mahalanobis":[], "cosine_sil":[], "mahalanobis_sil":[]}

if args.type == "wesad":
    train_ids = os.listdir(os.path.join(args.data_dir))
    train_dataset = WESADDataset(pkl_files=[os.path.join(args.data_dir, id, id+".pkl") for id in train_ids], test_mode=args.test)
    train_dataset, valid_data = random_split(train_dataset, [0.8, 0.2])
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn= lambda x:x)
if args.type == "kemo":
    train_dataset = KemoDataset(args.data_dir, test_mode=args.test)
    train_dataset, valid_data = random_split(train_dataset, [0.8, 0.2])
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn= lambda x:x)
def set_parameters(net, new_parameters):
    for old, new in zip(net.parameters(), new_parameters):
        shape = old.data.size()
        old.data = torch.Tensor(new).view(shape).to(DEVICE)
        
def client_fn(context: Context):
    net = GRU(3, 4, 1)
    net.to(DEVICE)
    rate = random.randint(2, 9)/10
    trainset, _ = random_split(train_dataset, [rate, 1-rate])
    train_loader = DataLoader(trainset, args.batch_size, shuffle=True , collate_fn=lambda x: x)
    if args.type == "wesad":
        return FedAvgClient(net, train_loader=train_loader, valid_loader=valid_loader, epoch=args.epoch, lossf=Lossf, optimizer=torch.optim.SGD(net.parameters(), lr= args.lr), trainF=wesad.train, validF=wesad.valid, DEVICE=DEVICE).to_client()
    elif args.type == "kemo":
        return FedAvgClient(net, train_loader=train_loader, valid_loader=valid_loader, epoch=args.epoch, lossf=Lossf, optimizer=torch.optim.SGD(net.parameters(), lr= args.lr), trainF=kemo.train, validF=kemo.valid, DEVICE=DEVICE).to_client()
    
if __name__=="__main__":
    warnings.filterwarnings("ignore")
    wesad.make_dir(os.path.join("./Models", args.version))

    def server_fn(context):
        if args.type == "wesad":
            net = GRU(3, 4, 1)
        elif args.type == "kemo":
            net = K_emo_GRU(3, 4, 1)
        net.to(DEVICE)
        if args.mode == "fedavg":
            strategy = FedAvgServer(net= net, testLoader=valid_loader, args=args,inplace=True, lossf=Lossf,min_fit_clients=10, min_available_clients=10, min_evaluate_clients=10)
        elif args.mode == "ccfl":
            strategy = ClusteredFedAvg(net=net, testLoader=valid_loader, args=args, inplace=False, lossf=Lossf, CosVsMaha=CosVsMaha,min_fit_clients=10, min_available_clients=10, min_evaluate_clients=10)
        elif args.mod == "mdcfl":
            strategy = ClusteredFedAvg(net=net, testLoader=valid_loader, args=args, inplace=False, lossf=Lossf, CosVsMaha=CosVsMaha,min_fit_clients=10, min_available_clients=10, min_evaluate_clients=10)
        return fl.server.ServerAppComponents(strategy=strategy, config=fl.server.ServerConfig(args.round))
    
    fl.simulation.run_simulation(
        client_app= fl.client.ClientApp(client_fn=client_fn),
        server_app= fl.server.ServerApp(server_fn=server_fn),
        num_supernodes = 10,
        backend_config={"client_resources": {"num_cpus": 1.0 , "num_gpus": 1}},
        verbose_logging=False
    )
    performance_cluster = pd.DataFrame(CosVsMaha, index=None)[1:]
    performance_cluster.to_csv(f"./Csv/{args.version}_cluster.csv")
from Network import *
from utils import *
import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split
from Network import GRU, K_emo_GRU
import numpy as np
import random
from torchmetrics import Accuracy, F1Score
import os
from flwr.common import Context
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from client.client import FedAvgClient
from server.NSMDClusteredFedAvgServer import ClusteredFedAvg, CosVsMaha
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
args = parser.Federatedparser()
set_seeds(args)
Accuracyf = Accuracy(task= "multiclass", num_classes=2)
F1Scoref = F1Score(task= "multiclass", num_classes=2)
Lossf = torch.nn.BCEWithLogitsLoss().to(DEVICE)

train_ids = os.listdir(os.path.join(args.wesad_path))
train_dataset = WESADDataset(pkl_files=[os.path.join(args.wesad_path, id, id+".pkl") for id in train_ids], test_mode=args.test)
train_dataset, valid_data = random_split(train_dataset, [0.8, 0.2])
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn= lambda x:x)

def set_parameters(net, new_parameters):
    for old, new in zip(net.parameters(), new_parameters):
        shape = old.data.size()
        old.data = torch.Tensor(new).view(shape).to(DEVICE)
        
def client_fn(context: Context):
    net = GRU(3, 4, 1)
    net.to(DEVICE)
    rate = random.randint(2, 10)/10
    trainset, _ = random_split(train_dataset, [rate, 1-rate])
    train_loader = DataLoader(trainset, args.batch_size, shuffle=True , collate_fn=lambda x: x)
    return FedAvgClient(net, train_loader=train_loader, valid_loader=valid_loader, epoch=args.epoch, lossf=Lossf, optimizer=torch.optim.SGD(net.parameters(), lr= args.lr), trainF=train, validF=valid, DEVICE=DEVICE).to_client()
class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *, net, testLoader, args, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn=None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures: bool = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace: bool = True) -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.net = net
        self.testLoader = testLoader
        self.args = args     
    def evaluate(self, server_round: int, parameters):
        parameters = parameters_to_ndarrays(parameters)
        lossf = nn.BCEWithLogitsLoss()
        set_parameters(self.net, parameters)
        history = valid(self.net, self.testLoader, 0, lossf, DEVICE)
        make_model_folder("./Result")
        make_model_folder(f"./Result/{self.args.version}")
        if server_round != 0:
            old_historyframe = pd.read_csv(os.path.join("./Result", self.args.version, f'MDCFL.csv'))
            historyframe = pd.DataFrame({k:[v] for k, v in history.items()})
            newframe=pd.concat([old_historyframe, historyframe])
            newframe.to_csv(os.path.join("./Result", self.args.version, f'MDCFL.csv'), index=False)
        else:
            pd.DataFrame({k:[v] for k, v in history.items()}).to_csv(os.path.join("./Result", self.args.version, f'MDCFL.csv'), index=False)
        torch.save(self.net.state_dict(), f"./Models/{self.args.version}/net.pt")
        return history['loss'], {key:value for key, value in history.items() if key != "loss" }

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    make_model_folder(os.path.join("./Models", args.version))
    
    def server_fn(context):
        net = GRU(3, 4, 1)
        net.to(DEVICE)
        # strategy = ClusteredFedAvg(net=net, testLoader=valid_loader, args=args, inplace=False)
        strategy = FedAvg(net= net, testLoader=valid_loader, args=args,inplace=True, min_fit_clients=10, min_available_clients=10, min_evaluate_clients=10)
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
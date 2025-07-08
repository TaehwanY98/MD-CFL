from typing import Dict, List, Optional, Tuple
import flwr 
import torch
from torch import save
from utils.Wesad_Train import valid as wesadValid, make_dir
from utils.Kemo_Train import valid as kemoValid
from Network import *

import os
import pandas as pd
from flwr.common import (
    parameters_to_ndarrays,
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FedAvgServer(flwr.server.strategy.FedAvg):
    def __init__(self, net, eval_loader, lossf, args):
        super().__init__()
        self.net = net
        self.eval_loader = eval_loader
        self.lossf = lossf
        self.args = args
    def aggregate_fit(self, ins: flwr.common.FitIns) -> flwr.common.FitRes:
        super().aggregate_fit(ins)
    def evaluate(self, server_round: int, parameters)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
        parameters = parameters_to_ndarrays(parameters)
        if self.args.type=="wesad":
            validF = wesadValid
        if self.args.type == "kemo":
            validF = kemoValid

        set_parameters(self.net, parameters)
        history=validF(self.net, self.validLoader, 0, self.lossf.to(DEVICE), DEVICE, True)
        make_dir(self.args.result_path)
        make_dir(os.path.join(self.args.result_path, self.args.mode))
        if server_round != 0:
            old_historyframe = pd.read_csv(os.path.join(self.args.result_path, self.args.mode, f'FedAvg_{self.args.type}.csv'))
            historyframe = pd.DataFrame({k:[v] for k, v in history.items()})
            newframe=pd.concat([old_historyframe, historyframe])
            newframe.to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedAvg_{self.args.type}.csv'), index=False)
        else:
            pd.DataFrame({k:[v] for k, v in history.items()}).to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedAvg_{self.args.type}.csv'), index=False)
        save(self.net.state_dict(), f"./Models/{self.args.version}/net.pt")
        return history['loss'], {key:value for key, value in history.items() if key != "loss" }
    
def set_parameters(net, parameters):
    for old, new in zip(net.parameters(), parameters):
        old.data = torch.Tensor(new).to(DEVICE)
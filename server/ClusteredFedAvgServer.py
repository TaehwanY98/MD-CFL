from collections import OrderedDict
from logging import WARNING
from logging import WARNING
from typing import Callable, Optional, Union, List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_distances
# from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import GraphicalLasso
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
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
import flwr 
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import aggregate, aggregate_inplace, weighted_loss_avg
from scipy.spatial.distance import mahalanobis
import torch
import torch.nn as nn
from torch import save
from torch.utils.data import DataLoader
# from Old_train import valid, make_model_folder
import warnings
from utils.Wesad_Train import valid as wesadValid, make_dir
from utils.Kemo_Train import valid as kemoValid
from Network import *
import os
import numpy as np
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


kmeans = KMeans(n_clusters=2)

def set_parameters(net, parameters):
    for old, new in zip(net.parameters(), parameters):
        old.data = torch.Tensor(new).to(DEVICE)
        
def make_model_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

def mahalanobis_distance_cal(X,Y, cov=None):
    # GL = GraphicalLasso()
    mahala = []
    for x, y in zip(X,Y):
        eye=np.eye(x.size)
        distance = mahalanobis(x.reshape(-1,), y.reshape(-1,), np.ones_like(eye)) if cov is None else mahalanobis(x.reshape(-1,), y.reshape(-1,), cov)
        if not np.isnan(distance):
            mahala.append(distance)
        else:
            mahala.append(np.abs(x.flatten()-y.flatten()).sum())
    return mahala

def make_inv_cov(eye, x, y):
    for xt in range(x.size):
        for yt in range(y.size):
            eye[xt, yt] = abs(x[xt]-y[yt])
    return eye.T

def cosine_distance_cal(X, Y):
    try:
        cosine = [cosine_distances(x.reshape(1, -1),y.reshape(1, -1))[0]  for x,y in zip(X,Y)]
    except:
        cosine = [cosine_distances(x,y)[0]  for x,y in zip(X,Y)]
    return [float(param[0]) for param in cosine]

def parameter_to_Ndarrays(param):
    return [v.flatten() for v in param]

def parameter_Dnumber_samples(param, num_examples, sum_of_exmaples):
    return [np.array(list(map(lambda x: x*num_examples/sum_of_exmaples, p))) for p in param]

class ClusteredFedAvg(flwr.server.strategy.FedAvg):
    def __init__(self, net, testLoader, args, lossf , CosVsMaha, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):    
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.net = net
        self.testLoader = testLoader
        self.args = args
        self.lossf = lossf
        self.CosVsMaha = CosVsMaha
    def aggregate_fit(self, server_round, results, failures):
        clusters = {}
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")
            
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            
            """Extract Parameter for distances """
            only_params = [parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results]
            
            aggregated_ndarrays = aggregate(weights_results)
            
            '''cosine Clustering Part'''
            if self.args.mode == "ccfl":
                for indx, client_params in  enumerate([parameter_to_Ndarrays(params) for params in only_params]):
                    clusters[f"client{indx+1}"] = cosine_distance_cal(client_params, aggregated_ndarrays)
            elif self.args.mode == "mdcfl":
                '''mahalanobis Clustering Part'''
                for indx, client_params in  enumerate([parameter_to_Ndarrays(params) for params in only_params]):
                    clusters[f"client{indx+1}"] = mahalanobis_distance_cal(client_params, aggregated_ndarrays)
            cluster_indexs = kmeans.fit_predict( [value for _, value in clusters.items()])

            print(cluster_indexs)
            if self.args.mode == "ccfl":
                self.CosVsMaha["cosine"].append(kmeans.inertia_)
            elif self.args.mode == "mdcfl":
                self.CosVsMaha["mahalanobis"].append(kmeans.inertia_)
                
                
            n1 = np.count_nonzero(cluster_indexs)
            n0 = len(cluster_indexs)-n1
            if n1==0 or n0 ==0:
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                return parameters_aggregated, metrics_aggregated
            else:
                if n1> n0:
                    indexs = np.arange(len(cluster_indexs))[cluster_indexs==1]
                else:
                    indexs = np.arange(len(cluster_indexs))[cluster_indexs==0]

                weights_results=[weights_results[i] for i in indexs]
                only_params=[parameter_to_Ndarrays(params) for params in only_params]
                aggregated_ndarrays_indexing = aggregate(weights_results)
                
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays_indexing)
                if self.args.mode == "ccfl":
                    self.CosVsMaha["cosine_sil"].append(sum([silhouette_score(np.stack(samples), cluster_indexs) for samples in zip(*only_params)])/len(list(zip(*only_params))))
                elif self.args.mode == "mdcfl":
                    self.CosVsMaha["mahalanobis_sil"].append(sum([silhouette_score(np.stack(samples), cluster_indexs) for samples in zip(*only_params)])/len(list(zip(*only_params))))

                if server_round != 0:
                    historyframe = pd.DataFrame(self.CosVsMaha)
                    historyframe.to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedAvg_{self.args.type}_sil.csv'), index=False)
                else:
                    pd.DataFrame(self.CosVsMaha).to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedAvg_{self.args.type}_sil.csv'), index=False)
                
                return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, server_round: int, parameters):
        parameters = parameters_to_ndarrays(parameters)
        if self.args.type=="wesad":
            validF = wesadValid
        if self.args.type == "kemo":
            validF = kemoValid

        set_parameters(self.net, parameters)
        history=validF(self.net, self.testLoader, 0, self.lossf.to(DEVICE), DEVICE)
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

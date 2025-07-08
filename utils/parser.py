import argparse

def Centralparser():
    parser= argparse.ArgumentParser(
        prog="Central Learning in WESAD",
        description="centralized training code by using WESAD Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-s", "--seed", type= int, default= 2024)
    parser.add_argument("-e", "--epoch", type= int, default= 10)
    parser.add_argument("-bs", "--batch-size", type= int, default=1)
    parser.add_argument("-l", "--lr", type= float, default= 1e-2)
    parser.add_argument("-w", "--wesad_path", type= str, default=None)
    parser.add_argument("-p", "--pretrained", type= str, default=None)
    args = parser.parse_args()
    return args

def Federatedparser():
    parser= argparse.ArgumentParser(
        prog="Federated Learning in WESAD",
        description="Federated Learning code by using WESAD Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-s", "--seed", type= int, default= 2024)
    parser.add_argument("-r", "--round", type= int, default=10)
    parser.add_argument("-e", "--epoch", type= int, default= 3)
    parser.add_argument("-i", "--id", type= int, default=1)
    parser.add_argument("-bs", "--batch-size", type= int, default=1)
    parser.add_argument("-l", "--lr", type= float, default=4e-3)
    parser.add_argument("-p", "--pretrained", type= str, default=None)
    parser.add_argument("-w", "--wesad_path", type= str, default=None)
    parser.add_argument("-t", "--test", type= bool, default=False)
    parser.add_argument("-m", "--mode", type= str, default="max")
    args = parser.parse_args()
    return args

def Simulationparser():
    parser= argparse.ArgumentParser(
        prog="Simulation in WESAD",
        description="Simulation code by using WESAD Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-s", "--seed", type= int, default= 2024)
    parser.add_argument("-r", "--round", type= int, default=10)
    parser.add_argument("-e", "--epoch", type= int, default= 3)
    parser.add_argument("-bs", "--batch-size", type= int, default=1)
    parser.add_argument("-l", "--lr", type= float, default=4e-3)
    parser.add_argument("-d", "--data-dir", type= str, default=None, required=True)
    parser.add_argument("-cd", "--client-dir", type= str, default=None, required=True)
    parser.add_argument("--test", type= bool, default=False)
    parser.add_argument("-m", "--mode", type= str, default="fedavg")
    parser.add_argument("-t", "--type", type= str, default="wesad")
    parser.add_argument("--result-path", type= str, default="Result")
    args = parser.parse_args()
    return args
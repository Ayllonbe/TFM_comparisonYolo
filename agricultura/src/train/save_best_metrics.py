import argparse
import torch
from pathlib import Path
from yolov5 import hubconf
import pandas as pd
import re
import json
import yaml


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path",    type=str, default="results/train/weights/best.pt", help="Source data directory")
    parser.add_argument("-r", "--results_path",  type=str, default="results/train/results.csv",     help="Path to the results file")
    parser.add_argument("-s", "--save_path",     type=str, default="results/metrics",               help="Target data directory")
    return parser.parse_args()


def load_best_epoch(args):
    """ load the best epoch corresponding to the saved models"""
    start_epoch, fitness_score = 0, 0.0
    # Configure data paths
    weights_path = Path(args.model_path).absolute()
    # needed to load checkpoint
    model = hubconf.custom(path=weights_path, device='cpu')
    checkpoint = torch.load(weights_path)
    epoch = checkpoint['epoch']
    print("Best Model loaded from epoch: ", epoch)
    if checkpoint['best_fitness'] is not None: # Make sure best_fitness is not None as no Best model was saved
        fitness_score = checkpoint['best_fitness'][0]

    return epoch, fitness_score


def load_results(args, epoch):
    """ load the training results for the saved best epoch """
    results_fn = Path(args.results_path).absolute()
    results = pd.read_csv(results_fn)
    best_results = results.iloc[epoch]
    return best_results


def save_best_metrics(best_results, val_acc, filename):
    """ Save the best training metrics dictionary as json file"""
    # Get the index but remove spaces and substitute special chars [/:.]
    metrics = dict(zip( [re.sub('[/:.]', '_', h.replace(" ", "")) for h in best_results.index], best_results))
    # Remove the learning rate metrics
    metrics = dict( [(k,v) for k,v in metrics.items() if not k.startswith('x_lr')] )
    # Add the extracted fitness_score as accuracy
    metrics["accuracy"] = val_acc
    metrics=dict(sorted(metrics.items()))

    # Save the metrics dictionary
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as fp:
        json.dump(metrics, fp)

def main(args):
    """ Store the best result metrics as json for DVC tracking from trained weights"""
    epoch , val_acc = load_best_epoch(args)
    best_results = load_results(args,epoch)
    fn = Path(args.save_path)/"train_metrics.json"
    save_best_metrics(best_results, val_acc, str(fn))

if __name__==  '__main__':
    args = args_parse()
    main(args)

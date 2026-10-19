#!/usr/bin/env python3
"""Count samples, windows, updates and model parameters without loading data."""
import argparse
import json
import math

from bearing_fault_cnn import BearingCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows-per-file', type=int, default=420000)
    parser.add_argument('--training-files', type=int, default=3)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=23)
    args = parser.parse_args()
    if min(vars(args).values()) < 1 or args.rows_per_file < args.length:
        parser.error('All counts must be positive; each file must contain at least one window')
    windows = 1+(args.rows_per_file-args.length)//args.stride
    examples = args.training_files*windows
    steps = math.ceil(examples/args.batch_size)
    model = BearingCNN()
    parameters = {name:p.numel() for name,p in model.named_parameters()}
    print(json.dumps({'raw_training_readings':args.rows_per_file*args.training_files,
                      'windows_per_file':windows,'training_examples':examples,
                      'positions_presented_per_epoch':examples*args.length,
                      'optimizer_updates_per_epoch':steps,'optimizer_updates_total':steps*args.epochs,
                      'example_presentations_total':examples*args.epochs,
                      'parameters_by_tensor':parameters,'total_parameters':sum(parameters.values()),
                      'weight_bytes_float32':4*sum(parameters.values()),
                      'note':'Window and batch arguments are counting scenarios; network architecture is the reference model. Overlap and repeated epochs do not create independent recordings. Weight bytes exclude gradients, optimizer state and activations.'},indent=2))


if __name__ == '__main__':
    main()

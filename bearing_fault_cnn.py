#!/usr/bin/env python3
"""Classify bearing signals using exactly two Conv1d layers (PyTorch).

Install:
    python3 -m pip install numpy torch matplotlib
Run (CSV files default to the directory containing this script):
    python3 bearing_fault_cnn.py
    python3 bearing_fault_cnn.py --data-dir /path/to/csvs --epochs 50

Only the first CSV column is used; numeric strings are converted to floats.
An optional header is detected. Invalid/missing signal values cause an error
instead of silently removing samples and changing the signal timing.

Train:      H_1_0.csv, I_1_2.csv, O_6_2.csv
Validation: H_3_0.csv, I_3_2.csv, O_7_2.csv
Test:       H_4_0.csv, I_4_2.csv, O_8_2.csv

"1024 segments" is interpreted as segments LENGTH 1024 samples. With 420,000
samples, each file supplies 410 non-overlapping segments; 160 trailing samples
are discarded. Creating 1,024 such segments would require overlap or padding.

The validation recording selects the best epoch. Test recordings are evaluated
only after training. Normalization statistics come exclusively from training.
Per-class test accuracy = correctly predicted samples of that class / all test
samples of that class (class recall), reported as a percentage. These results
describe segment classification on the three held-out recordings.
"""

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SIGNAL_LENGTH = 1024
BATCH_SIZE = 32
CLASS_NAMES = ('Healthy', 'Inner race', 'Outer race')
FILE_SPLITS = {
    'train': ('H_1_0.csv', 'I_1_2.csv', 'O_6_2.csv'),
    'validation': ('H_3_0.csv', 'I_3_2.csv', 'O_7_2.csv'),
    'test': ('H_4_0.csv', 'I_4_2.csv', 'O_8_2.csv'),
}


def read_signal(path, max_samples):
    """Read the first column without using any other sensor or metadata column."""
    values = []
    total_samples = 0
    first_row = True
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        preview = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(preview, delimiters=',;\t')
        except csv.Error:
            dialect = csv.excel
        for row_number, row in enumerate(csv.reader(handle, dialect), start=1):
            if not row or not row[0].strip():
                raise ValueError(f'{path.name}, row {row_number}: empty first column.')
            token = row[0].strip()
            try:
                value = float(token)
            except ValueError as exc:
                if first_row:  # Optional text column header.
                    first_row = False
                    continue
                raise ValueError(
                    f'{path.name}, row {row_number}: first column is not numeric.'
                ) from exc
            first_row = False
            if not np.isfinite(value):
                raise ValueError(f'{path.name}, row {row_number}: non-finite sample.')
            total_samples += 1
            if len(values) < max_samples:
                values.append(value)
    signal = np.asarray(values, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError(f'{path.name}: sample exceeds the float32 range.')
    if len(signal) < SIGNAL_LENGTH:
        raise ValueError(f'{path.name}: at least {SIGNAL_LENGTH} samples are required.')
    if total_samples < max_samples:
        print(f'WARNING: {path.name} has {total_samples:,} samples; using all of them.')
    return signal, total_samples


def load_split(data_dir, split_name, max_samples):
    arrays, labels, manifest = [], [], []
    for label, filename in enumerate(FILE_SPLITS[split_name]):
        signal, total_samples = read_signal(data_dir / filename, max_samples)
        segment_count = len(signal) // SIGNAL_LENGTH
        used_samples = segment_count * SIGNAL_LENGTH
        # PyTorch Conv1d expects (batch, channels=1, signal_length=1024).
        segments = signal[:used_samples].reshape(segment_count, 1, SIGNAL_LENGTH)
        arrays.append(segments)
        labels.append(np.full(segment_count, label, dtype=np.int64))
        record = {
            'file': filename,
            'class': CLASS_NAMES[label],
            'available_samples': total_samples,
            'selected_samples': len(signal),
            'segments': segment_count,
            'discarded_tail_samples': len(signal) - used_samples,
            'samples_beyond_limit': total_samples - len(signal),
        }
        manifest.append(record)
        print(f'{split_name:10s} {filename:12s}: {total_samples:,} rows, '
              f'{segment_count} segments, {len(signal) - used_samples} tail samples discarded')
    return np.concatenate(arrays), np.concatenate(labels), manifest


class BearingCNN(nn.Module):
    """Exactly two convolutional layers, followed by a three-class classifier."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),  # Convolution 1
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # Convolution 2
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(CLASS_NAMES)),
        )

    def forward(self, signals):
        # CrossEntropyLoss consumes logits directly; no softmax is needed here.
        return self.network(signals)


def make_loader(signals, labels, shuffle, seed):
    dataset = TensorDataset(torch.from_numpy(signals), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, drop_last=False, generator=generator)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count += len(labels)
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / count, 100.0 * correct / count


def confusion_matrix(truth, predictions):
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(matrix, (truth, predictions), 1)
    return matrix


def binary_matrix(matrix, label):
    tp = int(matrix[label, label])
    fn = int(matrix[label].sum()) - tp
    fp = int(matrix[:, label].sum()) - tp
    tn = int(matrix.sum()) - tp - fn - fp
    return np.asarray([[tn, fp], [fn, tp]])


def save_confusion_matrices(matrix, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    matrices = [('All three classes', matrix, CLASS_NAMES)]
    matrices.extend((f'{name} vs rest', binary_matrix(matrix, label), ('Rest', name))
                    for label, name in enumerate(CLASS_NAMES))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (title, values, labels) in zip(axes.flat, matrices):
        ax.imshow(values, cmap='Blues', vmin=0)
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel('Predicted condition')
        ax.set_ylabel('True condition')
        ax.set_title(title)
        for row, column in np.ndindex(values.shape):
            color = 'white' if values[row, column] > values.max() / 2 else 'black'
            ax.text(column, row, str(values[row, column]), ha='center', va='center', color=color)
    fig.savefig(output_dir / 'confusion_matrices.png', dpi=180)
    plt.close(fig)
    with (output_dir / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['True / Predicted', *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name, *row.tolist()])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10, help='Early-stopping patience on validation loss.')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--max-samples', type=int, default=420000, help='First N signal samples per file.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    args = parser.parse_args()
    if (args.epochs < 1 or args.patience < 1 or args.max_samples < SIGNAL_LENGTH
            or args.threads < 1 or args.learning_rate <= 0):
        parser.error('epochs, patience, threads and learning-rate must be positive; max-samples must be >= 1024.')
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir
                  else data_dir / 'bearing_cnn_results')
    missing = [name for files in FILE_SPLITS.values() for name in files
               if not (data_dir / name).is_file()]
    if missing:
        parser.error(f'Missing CSV files in {data_dir}: {", ".join(missing)}')
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu.')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('Apple MPS is unavailable; use --device cpu.')
    if args.device in ('cpu', 'cuda'):
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    print(f'Device: {device} | batch size: {BATCH_SIZE} | signal length: {SIGNAL_LENGTH}')
    x_train, y_train, train_manifest = load_split(data_dir, 'train', args.max_samples)
    x_val, y_val, val_manifest = load_split(data_dir, 'validation', args.max_samples)
    mean = float(x_train.mean(dtype=np.float64))
    std = float(x_train.std(dtype=np.float64))
    if std < 1e-12:
        raise ValueError('Training signal is constant; normalization is not possible.')
    # One global scaler preserves amplitude differences; no test/validation fitting.
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_val = ((x_val - mean) / std).astype(np.float32)
    train_loader = make_loader(x_train, y_train, True, args.seed)
    val_loader = make_loader(x_val, y_val, False, args.seed)

    model = BearingCNN().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_epoch, best_state, stale_epochs = float('inf'), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError('Non-finite loss; check the signals and learning rate.')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy_pct': train_acc,
                        'validation_loss': val_loss, 'validation_accuracy_pct': val_acc})
        print(f'Epoch {epoch:02d}/{args.epochs}: train loss={train_loss:.4f}, '
              f'acc={train_acc:.2f}% | val loss={val_loss:.4f}, acc={val_acc:.2f}%', flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f'Early stopping; restoring epoch {best_epoch}.')
                break
    model.load_state_dict(best_state)
    print(f'Using epoch {best_epoch}, selected by minimum validation loss ({best_loss:.4f}).')

    # Load/evaluate the held-out test recordings only after model selection.
    x_test, y_test, test_manifest = load_split(data_dir, 'test', args.max_samples)
    x_test = ((x_test - mean) / std).astype(np.float32)
    test_loader = make_loader(x_test, y_test, False, args.seed)
    predictions, probabilities = [], []
    model.eval()
    with torch.inference_mode():
        for signals, _ in test_loader:
            probs = torch.softmax(model(signals.to(device)), dim=1).cpu().numpy()
            probabilities.append(probs)
            predictions.append(probs.argmax(axis=1))
    predictions = np.concatenate(predictions)
    probabilities = np.concatenate(probabilities)
    matrix = confusion_matrix(y_test, predictions)
    overall_accuracy = 100.0 * float(np.trace(matrix)) / int(matrix.sum())
    print(f'\nOverall test accuracy: {overall_accuracy:.2f}%')
    per_class = {}
    for label, name in enumerate(CLASS_NAMES):
        support, correct = int(matrix[label].sum()), int(matrix[label, label])
        class_accuracy = 100.0 * correct / support
        per_class[name] = {'file': FILE_SPLITS['test'][label], 'correct': correct,
                           'total': support, 'accuracy_pct': class_accuracy,
                           'one_vs_rest_confusion_matrix': binary_matrix(matrix, label).tolist()}
        print(f'{name:12s} ({FILE_SPLITS["test"][label]}): '
              f'{class_accuracy:.2f}% ({correct}/{support})')
    print('\nConfusion matrix: rows=true, columns=predicted; order=Healthy, Inner race, Outer race')
    print(matrix)
    for label, name in enumerate(CLASS_NAMES):
        print(f'\n{name} vs rest: rows=true, columns=predicted; order=Rest, {name}')
        print(binary_matrix(matrix, label))

    results = {
        'overall_test_accuracy_pct': overall_accuracy,
        'per_class': per_class,
        'class_order': CLASS_NAMES,
        'confusion_matrix': matrix.tolist(),
        'best_epoch': best_epoch,
        'best_validation_loss': best_loss,
        'config': {'signal_length': SIGNAL_LENGTH, 'batch_size': BATCH_SIZE,
                   'seed': args.seed, 'max_samples': args.max_samples,
                   'max_epochs': args.epochs, 'patience': args.patience,
                   'learning_rate': args.learning_rate, 'device': str(device),
                   'normalization_mean': mean, 'normalization_std': std,
                   'torch_version': str(torch.__version__), 'numpy_version': np.__version__},
        'files': {'train': train_manifest, 'validation': val_manifest, 'test': test_manifest},
        'history': history,
    }
    (output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    torch.save({'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'config': results['config'], 'class_order': CLASS_NAMES,
                'file_splits': FILE_SPLITS, 'best_epoch': best_epoch},
               output_dir / 'bearing_cnn.pt')
    with (output_dir / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['file', 'segment_index_0based', 'start_sample_0based', 'true_class',
                         'predicted_class', 'prob_healthy', 'prob_inner_race', 'prob_outer_race'])
        segment_indices = {label: 0 for label in range(len(CLASS_NAMES))}
        for truth, prediction, probs in zip(y_test, predictions, probabilities):
            index = segment_indices[int(truth)]
            writer.writerow([FILE_SPLITS['test'][truth], index, index * SIGNAL_LENGTH,
                             CLASS_NAMES[truth], CLASS_NAMES[prediction], *probs.tolist()])
            segment_indices[int(truth)] += 1
    save_confusion_matrices(matrix, output_dir)
    print(f'\nSaved model, metrics, predictions and confusion matrices to: {output_dir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Inspect development signals without loading held-out test files.

Examples:
 python signal_diagnostics.py --demo --output-dir runs/synthetic-diagnostics
 python signal_diagnostics.py --data-dir data --split train --detrend constant --output-dir runs/train-diagnostics
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bearing_fault_cnn import FILE_SPLITS, SIGNAL_LENGTH, read_signal
from signal_processing import (add_preprocessing_arguments, amplitude_spectrum,
                               config_from_args, mean_psd, preprocess)


def summarize(windows):
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    return {'mean': float(windows.mean()), 'std': float(windows.std()),
            'rms_mean': float(rms.mean()),
            'peak_abs': float(np.abs(windows).max()),
            'crest_factor_mean': float(np.mean(np.max(np.abs(windows), axis=1) / np.maximum(rms, 1e-12)))}


def plot_comparison(raw, processed, config, destination, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    t = np.arange(raw.shape[-1]) / (config.sample_rate or 1.0)
    for data, label in ((raw, 'Input'), (processed, 'Processed')):
        axes[0].plot(t, data[0], label=label, alpha=.75, linewidth=.8)
        f, a = amplitude_spectrum(data[:1], config.sample_rate)
        axes[1].plot(f, a[0], label=label, linewidth=.8)
        f, p = mean_psd(data, config.sample_rate)
        axes[2].semilogy(f, np.maximum(p, 1e-18), label=label)
    axes[0].set(xlabel='Time (s)' if config.sample_rate else 'Sample index', ylabel='Signal (source units)', title=title)
    unit = 'Hz' if config.sample_rate else 'cycles/sample'
    axes[1].set(xlabel=f'Frequency ({unit})', ylabel='Peak amplitude', title='First-window Hann rFFT (not PSD)')
    axes[2].set(xlabel=f'Frequency ({unit})', ylabel=f'Signal units squared / {unit}', title='Mean Welch PSD across separately processed windows')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(destination, dpi=150, metadata={})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data-dir', type=Path)
    source.add_argument('--demo', action='store_true', help='Synthetic signal only; not bearing evidence')
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--max-samples', type=int, default=420000)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_preprocessing_arguments(parser)
    args = parser.parse_args()
    try:
        config = config_from_args(args)
        if args.max_samples < SIGNAL_LENGTH:
            raise ValueError('max-samples must be >= 1024')
        if args.output_dir.exists():
            raise ValueError('Use a new output directory; existing evidence is never overwritten')
        if args.demo:
            from signal_processing import PreprocessConfig
            config = PreprocessConfig(detrend='constant', sample_rate=1024, high_hz=100)
            rng = np.random.default_rng(42)
            t = np.arange(8 * SIGNAL_LENGTH) / 1024
            clean = np.sin(2 * np.pi * 40 * t)
            raw = clean + .7 * np.sin(2 * np.pi * 300 * t) + .2 * rng.normal(size=len(t)) + 2
            inputs = [('synthetic-40Hz-plus-interference', raw.reshape(-1, SIGNAL_LENGTH))]
        else:
            inputs = []
            for name in FILE_SPLITS[args.split]:
                raw, _ = read_signal(args.data_dir / name, args.max_samples)
                inputs.append((Path(name).stem, raw[:len(raw)//SIGNAL_LENGTH*SIGNAL_LENGTH].reshape(-1, SIGNAL_LENGTH)))
        args.output_dir.mkdir(parents=True)
        report = {'source': 'synthetic' if args.demo else args.split, 'config': config.to_dict(),
                  'frequency_unit': 'Hz' if config.sample_rate else 'cycles/sample', 'files': {}}
        for name, raw in inputs:
            processed = preprocess(raw, config)
            plot_comparison(raw, processed, config, args.output_dir / f'{name}.png', name)
            report['files'][name] = {'windows': len(raw), 'before': summarize(raw), 'after': summarize(processed)}
        (args.output_dir / 'diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

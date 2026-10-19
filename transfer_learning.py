"""Load a trusted source checkpoint with explicit compatibility and leakage gates."""
import numpy as np
import torch

from bearing_fault_cnn import CLASS_NAMES


def initialize_transfer(model, path, config, kind, target_manifest, freeze_backbone=False):
    # Never load arbitrary downloaded checkpoints. weights_only limits pickle
    # reconstruction but is not a guarantee that untrusted files are harmless.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    source_config = checkpoint['config']
    if tuple(checkpoint['class_order']) != CLASS_NAMES:
        raise ValueError('This example requires the same ordered three-class task')
    source_preprocessing = source_config.get('preprocessing', {
        'detrend':'none','sample_rate':None,'low_hz':None,'high_hz':None,
        'notch_hz':None,'notch_q':30.0,'order':4})
    if source_preprocessing != config.to_dict() or source_config.get('representation','time') != kind:
        raise ValueError('Source and target preprocessing/representation must match; FFT is not interchangeable with time input')
    expected_length = 1024 if kind == 'time' else 513
    if source_config.get('input_features',source_config.get('signal_length')) != expected_length:
        raise ValueError('Source feature length differs from target input')
    source_manifest = checkpoint.get('split_manifest',checkpoint.get('file_splits'))
    if source_manifest is None:
        raise ValueError('Source training/validation provenance is required')
    def names(rows):
        return {row if isinstance(row,str) else row['file'] for row in rows}
    sources = [source_manifest]
    ancestor = checkpoint.get('transfer')
    if ancestor:
        sources.extend(ancestor.get('ancestral_manifests',[]))
    source_used = set().union(*(names(s['train']) | names(s['validation']) for s in sources))
    target_holdout = names(target_manifest['validation']) | names(target_manifest['test'])
    if source_used & target_holdout:
        raise ValueError('Source training/selection overlaps target validation/test recordings')
    # Check physical identity too when both manifests carry it. Filename gates
    # alone cannot detect renamed duplicates or unknown physical bearings.
    source_records = [r for s in sources for part in ('train','validation') for r in s[part]]
    target_records = target_manifest['validation'] + target_manifest['test']
    source_groups = {r['group'] for r in source_records if isinstance(r,dict) and r.get('group')}
    target_groups = {r['group'] for r in target_records if r.get('group')}
    if source_groups & target_groups:
        raise ValueError('Source and target holdouts share physical group identities')
    mean, std = source_config['normalization_mean'], source_config['normalization_std']
    if not np.isfinite([mean,std]).all() or std < 1e-12:
        raise ValueError('Invalid source scaler')
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    if freeze_backbone:
        for layer in list(model.network.children())[:8]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    return float(mean), float(std), {
        'mode':'head-only' if freeze_backbone else 'fine-tune-all',
        'normalization':'reuse source scaler unchanged',
        'source_best_epoch':checkpoint['best_epoch'],
        'ancestral_manifests':sources,
        'group_check':'performed for available group IDs; missing identity remains unverified',
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters':sum(p.numel() for p in model.parameters())}

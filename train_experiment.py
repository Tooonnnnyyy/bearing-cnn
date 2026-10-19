#!/usr/bin/env python3
"""Optional preprocessing experiment; default runs train/validation only.

Reference bearing_fault_cnn.py stays byte-identical. Inputs contain 1024 raw
samples; log-spectrum inputs contain 513 one-sided frequency bins. This still
uses the same two Conv1d layers. A new output directory is required for each run.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

import bearing_fault_cnn as baseline
from data_splits import validate_records
from signal_processing import add_preprocessing_arguments, config_from_args, preprocess, representation


def read_manifest(path=None):
    if path is None:
        return {split:[{'file':name, 'label':label} for label,name in enumerate(names)]
                for split,names in baseline.FILE_SPLITS.items()}
    payload = json.loads(path.read_text())
    manifest = payload['files']
    all_records = [record for split in ('train','validation','test') for record in manifest[split]]
    validate_records(all_records)
    for split in ('train','validation','test'):
        if {r['label'] for r in manifest[split]} != {0,1,2}:
            raise ValueError(f'{split} requires all three classes')
    return manifest


def load(data_dir, records, maximum, config, kind):
    arrays, labels, evidence = [], [], []
    for record in records:
        path = data_dir / record['file']
        values, count = baseline.read_signal(path, maximum)
        length = len(values)//baseline.SIGNAL_LENGTH*baseline.SIGNAL_LENGTH
        windows = values[:length].reshape(-1, baseline.SIGNAL_LENGTH)
        features = representation(preprocess(windows, config), kind)
        arrays.append(features[:,None,:])
        labels.append(np.full(len(windows),record['label'],dtype=np.int64))
        evidence.append({'file':record['file'], 'label':record['label'], 'available_samples':count,
                         'windows':len(windows), 'discarded_tail':len(values)-length,
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return np.concatenate(arrays), np.concatenate(labels), evidence


def evaluate(model, loader):
    truths, predictions = [], []
    model.eval()
    with torch.inference_mode():
        for features, labels in loader:
            predictions.extend(model(features).argmax(1).tolist())
            truths.extend(labels.tolist())
    matrix = baseline.confusion_matrix(np.asarray(truths),np.asarray(predictions))
    recall = np.diag(matrix)/matrix.sum(1)
    precision = np.divide(np.diag(matrix),matrix.sum(0),out=np.zeros(3,dtype=float),where=matrix.sum(0)>0)
    f1 = np.divide(2*precision*recall, precision+recall,out=np.zeros(3),where=precision+recall>0)
    return {'accuracy_pct':float(100*np.trace(matrix)/matrix.sum()),
            'balanced_accuracy_pct':float(100*recall.mean()), 'macro_f1_pct':float(100*f1.mean()),
            'class_order':list(baseline.CLASS_NAMES), 'recall_pct':(100*recall).tolist(),
            'precision_pct':(100*precision).tolist(), 'confusion_matrix':matrix.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path)
    p.add_argument('--representation', choices=('time','log-spectrum'), default='time')
    p.add_argument('--pretrained', type=Path, help='Trusted local checkpoint with source preprocessing and split provenance')
    p.add_argument('--freeze-backbone', action='store_true', help='With --pretrained: train only dense classifier parameters')
    p.add_argument('--unfreeze-after', type=int, default=0, help='If >0, unfreeze after N head-only epochs and reset Adam at 0.1x learning rate')
    p.add_argument('--evaluate-test', action='store_true', help='Opt in only after freezing the development choice')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-samples', type=int, default=420000)
    p.add_argument('--learning-rate', type=float, default=.001)
    add_preprocessing_arguments(p)
    args = p.parse_args()
    try:
        config = config_from_args(args)
        manifest = read_manifest(args.split_manifest)
        if args.freeze_backbone and args.pretrained is None:
            raise ValueError('--freeze-backbone requires --pretrained')
        if args.unfreeze_after < 0 or args.unfreeze_after >= args.epochs:
            raise ValueError('unfreeze-after must be nonnegative and below epochs')
        if args.unfreeze_after and not args.freeze_backbone:
            raise ValueError('--unfreeze-after requires --freeze-backbone')
        if args.output_dir.exists():
            raise ValueError('Use a new output directory to preserve previous evidence')
        if min(args.epochs,args.patience,args.threads) < 1 or args.max_samples < 1024 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError('epochs/patience/threads/rate must be positive; max-samples >=1024')
    except (KeyError,ValueError) as exc:
        p.error(str(exc))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    x, y, train_evidence = load(args.data_dir,manifest['train'],args.max_samples,config,args.representation)
    xv, yv, val_evidence = load(args.data_dir,manifest['validation'],args.max_samples,config,args.representation)
    model = baseline.BearingCNN()
    transfer = None
    if args.pretrained:
        from transfer_learning import initialize_transfer
        mean, std, transfer = initialize_transfer(model,args.pretrained,config,args.representation,manifest,args.freeze_backbone)
        transfer['checkpoint_sha256'] = hashlib.sha256(args.pretrained.read_bytes()).hexdigest()
    else:
        mean, std = float(x.mean()), float(x.std())
    if std < 1e-12:
        raise ValueError('Training features are constant')
    def scaled_loader(values, labels, shuffle=False):
        scaled = ((values-mean)/std).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError('Non-finite normalized features')
        return baseline.make_loader(scaled, labels, shuffle, args.seed)
    train_loader, val_loader = scaled_loader(x,y,True), scaled_loader(xv,yv)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best, best_epoch, stale, history = float('inf'), 0, 0, []
    for epoch in range(1,args.epochs+1):
        if args.unfreeze_after and epoch == args.unfreeze_after + 1:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate * .1)
            stale = 0
            transfer['mode'] = 'head-then-fine-tune'
            transfer['unfreeze_epoch'] = epoch
            transfer['fine_tune_learning_rate'] = args.learning_rate * .1
            transfer['final_trainable_parameters'] = sum(p.numel() for p in model.parameters())
        train_loss, train_acc = baseline.run_epoch(model,train_loader,criterion,torch.device('cpu'),optimizer)
        val_loss, val_acc = baseline.run_epoch(model,val_loader,criterion,torch.device('cpu'))
        if not np.isfinite([train_loss,val_loss]).all():
            raise ValueError('Non-finite loss')
        history.append({'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss,
                        'train_accuracy_pct':train_acc,'validation_accuracy_pct':val_acc})
        print(f'Epoch {epoch}: train={train_acc:.2f}% validation={val_acc:.2f}%',flush=True)
        if val_loss < best:
            best, best_epoch, state, stale = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience and (not args.unfreeze_after or epoch > args.unfreeze_after):
                break
    model.load_state_dict(state)
    settings = {'preprocessing':config.to_dict(),'representation':args.representation,
                'raw_length':1024,'input_features':int(x.shape[-1]),'batch_size':32,
                'normalization_mean':mean,'normalization_std':std,
                'normalization_fit':'source train only' if transfer else 'target train only',
                'seed':args.seed,'max_epochs':args.epochs,'patience':args.patience,'learning_rate':args.learning_rate,
                'unfreeze_after':args.unfreeze_after,
                'max_samples':args.max_samples,'threads':args.threads,'device':'cpu',
                'torch_version':str(torch.__version__),'numpy_version':np.__version__}
    import scipy
    settings['scipy_version'] = scipy.__version__
    results = {'config':settings,'best_epoch':best_epoch,'history':history,
               'files':{'train':train_evidence,'validation':val_evidence},
               'validation':evaluate(model,val_loader),'test':None,
               'test_note':'Not loaded or evaluated. Development evidence only.', 'transfer':transfer}
    args.output_dir.mkdir(parents=True)
    torch.save({'model_state_dict':state,'config':settings,'class_order':baseline.CLASS_NAMES,
                'split_manifest':manifest,'best_epoch':best_epoch, 'transfer':transfer},args.output_dir/'model.pt')
    if args.evaluate_test:
        xt, yt, evidence = load(args.data_dir,manifest['test'],args.max_samples,config,args.representation)
        results['test'] = evaluate(model,scaled_loader(xt,yt))
        results['files']['test'] = evidence
        results['test_note'] = 'Opt-in evaluation; repeated test-guided selection requires new independent test data.'
        baseline.save_confusion_matrices(np.array(results['test']['confusion_matrix']),args.output_dir)
    (args.output_dir/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'validation':results['validation'],'test':results['test']},indent=2))


if __name__ == '__main__':
    main()

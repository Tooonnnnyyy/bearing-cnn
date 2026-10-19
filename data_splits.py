#!/usr/bin/env python3
"""Create recording-level split manifests from explicit metadata.

Metadata is a JSON list of {file, label, group, domain, time}. label is 0/1/2;
group identifies the physical bearing when known; domain is an acquisition
condition; time is a sortable numeric timestamp/order. Do not infer these from
file numbers. Output is local experiment metadata, not a public data export.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split


METHODS = ('random', 'stratified', 'group', 'stratified-group', 'temporal', 'cross-domain')


def validate_records(records):
    if len(records) < 3:
        raise ValueError('Need at least three records and enough class/group coverage')
    files = []
    for record in records:
        name = record['file']
        if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name or not name.endswith('.csv'):
            raise ValueError('file must be a CSV basename, not an absolute or nested path')
        if type(record['label']) is not int or record['label'] not in (0, 1, 2):
            raise ValueError('label must be 0 (healthy), 1 (inner) or 2 (outer)')
        files.append(name)
    if len(set(files)) != len(files):
        raise ValueError('Each recording must appear exactly once')


def make_split(records, method='stratified-group', seed=42, test_size=.2,
               validation_size=.2, folds=3, fold=0, gap_records=0,
               validation_domain=None, test_domain=None):
    validate_records(records)
    if method not in METHODS:
        raise ValueError('Unknown split method')
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError('Split fractions must be positive and leave training data')
    indices = np.arange(len(records))
    y = np.array([r['label'] for r in records])
    if method in ('random', 'stratified'):
        strat = y if method == 'stratified' else None
        development, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=strat)
        train, validation = train_test_split(development, test_size=validation_size/(1-test_size),
                                            random_state=seed, stratify=y[development] if strat is not None else None)
    elif method == 'group':
        if any(not isinstance(r.get('group'), str) or not r['group'] for r in records):
            raise ValueError('Every record needs a nonempty group identity')
        groups = np.array([r['group'] for r in records])
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        development, test = next(splitter.split(indices, y, groups))
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size/(1-test_size), random_state=seed)
        a, b = next(splitter.split(development, y[development], groups[development]))
        train, validation = development[a], development[b]
    elif method == 'stratified-group':
        if any(not isinstance(r.get('group'), str) or not r['group'] for r in records):
            raise ValueError('Every record needs a nonempty group identity')
        if folds < 3 or not 0 <= fold < folds:
            raise ValueError('Use >=3 folds and a fold index in range')
        groups = np.array([r['group'] for r in records])
        for label in (0, 1, 2):
            if len(set(groups[y == label])) < folds:
                raise ValueError('Each class needs at least folds distinct groups; collect more data or reduce folds')
        assignments = list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(indices, y, groups))
        test = assignments[fold][1]
        validation = assignments[(fold + 1) % folds][1]
        train = np.setdiff1d(indices, np.concatenate([test, validation]))
    elif method == 'temporal':
        times = np.array([r['time'] for r in records], dtype=float)
        if not np.isfinite(times).all() or len(np.unique(times)) != len(times):
            raise ValueError('Use finite, unique recording times; group simultaneous acquisitions before splitting')
        if not isinstance(gap_records, int) or gap_records < 0:
            raise ValueError('gap_records must be a nonnegative integer')
        ordered = indices[np.argsort(times)]
        n_test, n_val = int(np.ceil(len(indices)*test_size)), int(np.ceil(len(indices)*validation_size))
        test_start, val_end = len(indices)-n_test, len(indices)-n_test-gap_records
        val_start = val_end-n_val
        if val_start-gap_records < 1:
            raise ValueError('Too few recordings for fractions and temporal gaps')
        train, validation, test = ordered[:val_start-gap_records], ordered[val_start:val_end], ordered[test_start:]
    else:
        if any(not isinstance(r.get('domain'), str) or not r['domain'] for r in records):
            raise ValueError('Every record needs a nonempty domain identity')
        if validation_domain is None or test_domain is None or validation_domain == test_domain:
            raise ValueError('Specify distinct validation_domain and test_domain')
        domains = np.array([r['domain'] for r in records])
        validation = indices[domains == validation_domain]
        test = indices[domains == test_domain]
        train = indices[(domains != test_domain) & (domains != validation_domain)]
    partitions = {'train':train, 'validation':validation, 'test':test}
    files, audit = {}, {}
    for name, rows in partitions.items():
        if not len(rows) or set(y[rows]) != {0, 1, 2}:
            raise ValueError(f'{name} lacks one or more classes; this design is infeasible for these records')
        files[name] = [records[int(i)] for i in rows]
        audit[name] = {'recordings':len(rows), 'class_counts':{str(label):int(np.sum(y[rows] == label)) for label in (0,1,2)}}
    overlaps = {}
    for a, b in (('train','validation'),('train','test'),('validation','test')):
        assert set(partitions[a]).isdisjoint(partitions[b])
        if all('group' in r for r in records):
            overlap = set(r['group'] for r in files[a]) & set(r['group'] for r in files[b])
            overlaps[f'{a}/{b}'] = len(overlap)
            if overlap and method in ('group','stratified-group'):
                raise ValueError('Group separation failed')
    used = set(np.concatenate(list(partitions.values())).tolist())
    return {'method':method, 'seed':seed, 'files':files, 'audit':audit,
            'group_overlap_counts':overlaps, 'excluded_records':len(records)-len(used),
            'note':'Class counts are recordings, not windows. Cross-domain separation does not imply unseen physical bearings; inspect group overlap.'}


def demo_records():
    return [{'file':f'synthetic_{domain}_{label}_{i}.csv', 'label':label,
             'group':f'bearing_{domain}_{label}_{i}', 'domain':domain,
             'time':domain_index*100 + i*3 + label}
            for domain_index, domain in enumerate(('A','B','C')) for label in range(3) for i in range(12)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=METHODS, default='stratified-group')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-size', type=float, default=.2)
    p.add_argument('--validation-size', type=float, default=.2)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--gap-records', type=int, default=0)
    p.add_argument('--validation-domain')
    p.add_argument('--test-domain')
    args = p.parse_args()
    try:
        if args.output.exists():
            raise ValueError('Use a new output path')
        records = json.loads(args.metadata.read_text())
        kwargs = {key:value for key,value in vars(args).items() if key not in ('metadata','output')}
        result = make_split(records, **kwargs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({key:value for key,value in result.items() if key != 'files'}, indent=2))
    except (KeyError, ValueError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()

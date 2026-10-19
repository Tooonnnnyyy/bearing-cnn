"""Scientific invariants, leakage guards and development-only execution."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import bearing_fault_cnn as baseline
from data_splits import demo_records, make_split
from signal_processing import PreprocessConfig, amplitude_spectrum, mean_psd, preprocess, representation
from transfer_learning import initialize_transfer


class SignalTests(unittest.TestCase):
    def test_fft_scaling_frequency_and_psd(self):
        n, fs = 1024, 1024
        t = np.arange(n)/fs
        x = (2*np.sin(2*np.pi*40*t))[None,:]
        f, amplitude = amplitude_spectrum(x,fs)
        self.assertEqual(f[amplitude[0].argmax()],40)
        self.assertAlmostEqual(amplitude[0,40],2,places=8)
        f, amplitude = amplitude_spectrum(np.ones((1,n))*3,fs,taper=False)
        self.assertEqual(amplitude[0,0],3)
        _, amplitude = amplitude_spectrum(((-1.)**np.arange(n))[None,:],fs,taper=False)
        self.assertEqual(amplitude[0,-1],1)
        f, p = mean_psd(x,fs)
        self.assertAlmostEqual(float(np.trapezoid(p,f)),2,places=3)
        self.assertEqual(representation(x,'log-spectrum').shape,(1,513))

    def test_filter_preserves_tone_and_attenuates_interference(self):
        t = np.arange(4096)/1024
        clean = np.sin(2*np.pi*40*t)
        raw = clean + .8*np.sin(2*np.pi*300*t)
        filtered = preprocess(raw[None,:],PreprocessConfig(sample_rate=1024,high_hz=100))[0]
        error_before = np.mean((raw[200:-200]-clean[200:-200])**2)
        error_after = np.mean((filtered[200:-200]-clean[200:-200])**2)
        self.assertLess(error_after,error_before*.01)
        for config in (PreprocessConfig(high_hz=100),PreprocessConfig(sample_rate=1024,high_hz=512),
                       PreprocessConfig(sample_rate=float('nan'))):
            with self.assertRaises(ValueError):
                preprocess(raw[None,:],config)

    def test_window_locality_and_detrending(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(2,1024))+np.arange(1024)*.1
        config = PreprocessConfig(detrend='linear',sample_rate=1024,high_hz=100)
        together = preprocess(x,config)
        np.testing.assert_allclose(together[0],preprocess(x[:1],config)[0])
        np.testing.assert_allclose(together[1],preprocess(x[1:],config)[0])
        residual = preprocess((2+np.arange(1024)*.1)[None,:],PreprocessConfig(detrend='linear'))
        self.assertLess(np.abs(residual).max(),1e-10)


class SplitTests(unittest.TestCase):
    def test_all_strategies_and_audit(self):
        records = demo_records()
        for method in ('random','stratified','group','stratified-group','temporal','cross-domain'):
            with self.subTest(method=method):
                result = make_split(records,method,validation_domain='B',test_domain='C',gap_records=1)
                groups = [{r['file'] for r in result['files'][s]} for s in ('train','validation','test')]
                self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
                for part in result['files'].values():
                    self.assertEqual({r['label'] for r in part},{0,1,2})
                if method == 'temporal':
                    self.assertEqual(result['excluded_records'],2)
                    self.assertLess(max(r['time'] for r in result['files']['train']), min(r['time'] for r in result['files']['validation']))
                if method == 'cross-domain':
                    self.assertEqual({r['domain'] for r in result['files']['test']},{'C'})
        # Repeated records of the same physical bearing must remain together.
        repeated = records + [dict(r,file='repeat_'+r['file']) for r in records]
        for method in ('group','stratified-group'):
            self.assertTrue(all(n==0 for n in make_split(repeated,method)['group_overlap_counts'].values()))

    def test_infeasible_and_duplicate_records_rejected(self):
        records = demo_records()
        with self.assertRaises(ValueError):
            make_split(records+[records[0]])
        with self.assertRaises(ValueError):
            make_split(records[:6])
        with self.assertRaises(ValueError):
            make_split(records,'cross-domain',validation_domain='A',test_domain='A')


class ExperimentTests(unittest.TestCase):
    def test_development_run_never_needs_test_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train','validation'):
                for label,name in enumerate(baseline.FILE_SPLITS[split]):
                    values = np.sin(np.arange(2048)*(label+1)*.03)+label
                    (root/name).write_text('Signal,Ignored\n'+'\n'.join(f'{v},ignored' for v in values)+'\n')
            cmd = [sys.executable,str(Path(baseline.__file__).with_name('train_experiment.py')),
                   '--data-dir',str(root),'--output-dir',str(root/'run'),'--epochs','1',
                   '--threads','2','--max-samples','2048','--detrend','constant','--representation','log-spectrum']
            subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=90)
            result = json.loads((root/'run/results.json').read_text())
            self.assertIsNone(result['test'])
            self.assertEqual(result['config']['input_features'],513)
            self.assertEqual(sum(map(sum,result['validation']['confusion_matrix'])),6)
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)

    def test_transfer_freezing_compatibility_and_source_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.pt'
            model = baseline.BearingCNN()
            config = PreprocessConfig()
            manifest = {s:[{'file':f'{s}_{i}.csv','label':i,'group':f'{s}_{i}'} for i in range(3)] for s in ('train','validation','test')}
            source = {s:[dict(r,file='source_'+r['file'],group='source_'+r['group']) for r in rows] for s,rows in manifest.items()}
            checkpoint = {'config':{'preprocessing':config.to_dict(),'representation':'time','input_features':1024,
                                    'normalization_mean':0.,'normalization_std':2.},
                          'model_state_dict':model.state_dict(),'class_order':baseline.CLASS_NAMES,
                          'split_manifest':source,'best_epoch':1}
            torch.save(checkpoint,path)
            mean,std,info = initialize_transfer(model,path,config,'time',manifest,True)
            self.assertEqual(std,2)
            self.assertFalse(model.network[0].weight.requires_grad)
            self.assertTrue(model.network[8].weight.requires_grad)
            before = model.network[0].weight.detach().clone()
            loss = model(torch.randn(3,1,1024)).square().mean()
            loss.backward()
            torch.optim.Adam((p for p in model.parameters() if p.requires_grad)).step()
            torch.testing.assert_close(model.network[0].weight,before)
            with self.assertRaises(ValueError):
                initialize_transfer(baseline.BearingCNN(),path,config,'log-spectrum',manifest)
            checkpoint['split_manifest'] = manifest
            torch.save(checkpoint,path)
            with self.assertRaises(ValueError):
                initialize_transfer(baseline.BearingCNN(),path,config,'time',manifest)

    def test_staged_transfer_cli_with_custom_split_and_opt_in_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = {s:[{'file':f'{s}_{i}.csv','label':i,'group':f'target_{s}_{i}'} for i in range(3)] for s in ('train','validation','test')}
            source = {s:[dict(r,file='source_'+r['file'],group='source_'+r['group']) for r in rows] for s,rows in target.items()}
            for rows in target.values():
                for record in rows:
                    x = np.sin(np.arange(1024)*.1*(record['label']+1))+record['label']
                    np.savetxt(root/record['file'],x)
            (root/'split.json').write_text(json.dumps({'files':target}))
            torch.save({'model_state_dict':baseline.BearingCNN().state_dict(),
                        'class_order':baseline.CLASS_NAMES,'split_manifest':source,'best_epoch':1,
                        'config':{'preprocessing':PreprocessConfig().to_dict(),'representation':'time',
                                  'input_features':1024,'normalization_mean':0.,'normalization_std':1.}},root/'source.pt')
            subprocess.run([sys.executable,str(Path(baseline.__file__).with_name('train_experiment.py')),
                            '--data-dir',str(root),'--output-dir',str(root/'output'),'--split-manifest',str(root/'split.json'),
                            '--pretrained',str(root/'source.pt'),'--freeze-backbone','--unfreeze-after','1',
                            '--epochs','2','--max-samples','1024','--threads','2','--evaluate-test'],
                           check=True,capture_output=True,text=True,timeout=90)
            result = json.loads((root/'output/results.json').read_text())
            self.assertEqual(result['transfer']['mode'],'head-then-fine-tune')
            self.assertEqual(result['transfer']['unfreeze_epoch'],2)
            self.assertAlmostEqual(result['transfer']['fine_tune_learning_rate'],.0001)
            self.assertEqual(sum(map(sum,result['test']['confusion_matrix'])),3)
            self.assertEqual(result['config']['normalization_fit'],'source train only')
            # Ancestors remain a leakage gate even after an intermediate run.
            model = baseline.BearingCNN()
            conflicting = {s:[dict(r,file='new_'+r['file'],group='new_'+r['group']) for r in rows] for s,rows in target.items()}
            conflicting['test'][0] = dict(source['train'][0])
            with self.assertRaises(ValueError):
                initialize_transfer(model,root/'output/model.pt',PreprocessConfig(),'time',conflicting)


if __name__ == '__main__':
    unittest.main()

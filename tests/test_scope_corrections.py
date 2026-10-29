from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import yaml

from modelaudit.artifacts import fingerprint
from modelaudit.audit_config import load_audit_config
from modelaudit.cli_experiments import run_cli_experiments, assess_report, synthetic_frame, scenario_inputs
from modelaudit.config import ConfigError
from modelaudit.data_checks import DataSchema, CheckPolicy, audit_data
from modelaudit.decisions import summarize_results
from modelaudit.drift import compare_snapshots
from modelaudit.predictions import positive_probabilities
from modelaudit.runner import run_audit
from modelaudit.scenarios import build_scenarios
from modelaudit.segments import evaluate_segments
import test_phase7


class PredictionTests(unittest.TestCase):
    def test_reversed_class_order_selects_positive_without_fit(self):
        model=Mock(classes_=np.array([1,0]))
        model.predict_proba.return_value=np.array([[.8,.2],[.1,.9]])
        frame=pd.DataFrame({'x':[1,2]})
        np.testing.assert_equal(positive_probabilities(model,frame),[.8,.1])
        model.fit.assert_not_called()
        pd.testing.assert_frame_equal(frame,pd.DataFrame({'x':[1,2]}))

    def test_invalid_other_class_probability_and_sum_rejected(self):
        for values in ([[.2,float('nan')],[.3,.7]],[[.2,1.1],[.3,.7]],[[.2,.2],[.3,.7]],[[.2],[.3]], [['.2','.8'],['.3','.7']]):
            model=Mock(classes_=np.array([0,1]));model.predict_proba.return_value=np.array(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                positive_probabilities(model,pd.DataFrame({'x':[1,2]}))

    def test_invalid_classes_rejected_before_prediction(self):
        for labels in ([0],[0,1,2],['no','yes'],[1,1]):
            model=Mock(classes_=np.array(labels))
            with self.subTest(labels=labels),self.assertRaises(ValueError):
                positive_probabilities(model,pd.DataFrame({'x':[1]}))
            model.predict_proba.assert_not_called()


class ScopeMetricTests(unittest.TestCase):
    def frame(self,prefix):
        return pd.DataFrame({'id':[f'{prefix}{i}' for i in range(80)],'x':np.arange(80),
                             'g':['a','b','b','a']*20,'y':[0,1]*40})

    def test_current_without_labels_and_prohibited_feature(self):
        train,ref=self.frame('T'),self.frame('R')
        current=self.frame('C').drop(columns='y').assign(forbidden=1)
        result=audit_data(train,ref,DataSchema(('x',),('g',),'y','id'),CheckPolicy(prohibited_columns=['forbidden']),
                          current=current,inspect_extra_columns=True)
        findings=[f for f in result['findings'] if f['dataset']=='current']
        self.assertTrue(any(f['check']=='prohibited_columns' and f['status']=='fail' for f in findings))
        self.assertFalse(any('y' in f['evidence'].get('missing_columns',[]) for f in findings))
        self.assertTrue(any(f['check']=='target' and f['status']=='skipped' for f in findings))
        self.assertNotIn('y',current)
        self.assertEqual(result,json.loads(json.dumps(result)))

    def test_invalid_prohibited_lists_rejected(self):
        for value in ('name',['x','x'],[1],['']):
            with self.subTest(value=value),self.assertRaises(ValueError): CheckPolicy(prohibited_columns=value)

    def test_audit_only_leakage_and_nonfinite_extra(self):
        train,ref=self.frame('T'),self.frame('R')
        ref['copy']=ref.y
        result=audit_data(train,ref,DataSchema(('x',),('g',),'y','id'),inspect_extra_columns=True)
        self.assertTrue(any('target_copy' in f['evidence'].get('signals',[]) for f in result['findings']))
        ref['copy']=np.inf
        result=audit_data(train,ref,DataSchema(('x',),('g',),'y','id'),inspect_extra_columns=True)
        self.assertTrue(any(f['status']=='fail' and f['evidence'].get('nonfinite_counts') for f in result['findings']))

    def test_single_class_reference_warns_while_train_fails(self):
        train,ref=self.frame('T'),self.frame('R');ref.y=0
        result=audit_data(train,ref,DataSchema(('x',),('g',),'y','id'))
        self.assertEqual(next(f['status'] for f in result['findings'] if f['check']=='target' and f['dataset']=='reference'),'warn')
        train.y=0
        result=audit_data(train,ref,DataSchema(('x',),('g',),'y','id'))
        self.assertEqual(next(f['status'] for f in result['findings'] if f['check']=='target' and f['dataset']=='train'),'fail')

    def test_best_group_direction_ties_and_support(self):
        labels=np.tile([0,1],80)
        scores=np.r_[np.tile([.1,.9],40),np.tile([.9,.1],40)]
        result=evaluate_segments(labels,scores,pd.DataFrame({'group':['good']*80+['bad']*80}))['segments'][0]
        groups={g['value']:g for g in result['groups']}
        self.assertEqual(result['best_by_metric']['recall']['groups'],['good'])
        self.assertEqual(groups['bad']['difference_from_best']['recall'],-1)
        self.assertGreater(groups['bad']['difference_from_best']['brier_score'],0)
        equal=evaluate_segments(labels,np.tile([.1,.9],80),pd.DataFrame({'group':['a']*80+['b']*80}))['segments'][0]
        self.assertEqual(equal['best_by_metric']['recall']['groups'],['a','b'])
        sparse=evaluate_segments([0,1],[.1,.9],pd.DataFrame({'group':['a','b']}))['segments'][0]
        self.assertIsNone(sparse['best_by_metric']['recall']['value'])
        self.assertTrue(all(g['difference_from_best']['recall'] is None for g in sparse['groups']))

    def test_jensen_shannon_extremes_and_known_value(self):
        def measure(a,b):
            return compare_snapshots(pd.DataFrame({'g':a}),pd.DataFrame({'g':b}),categorical=['g'],smoothing=0)['features']['g']
        self.assertEqual(measure(['a']*4,['a']*4)['jensen_shannon_distance'],0)
        self.assertEqual(measure(['a']*4,['b']*4)['jensen_shannon_distance'],1)
        result=measure(['a','a','a','b'],['a','b','b','b'])
        self.assertAlmostEqual(result['jensen_shannon_distance'],np.sqrt(.75*np.log2(1.5)+.25*np.log2(.5)))
        self.assertEqual(result['total_variation'],.5)

    def test_saved_scenario_suite_detects_falsely_passing_decisions(self):
        train,ref=test_phase7.frame(80,'T',20),test_phase7.frame(40,'R',30)
        predictions={n:np.where(ref.Churn==1,.8,.2) for n in ('logistic_regression','random_forest')}
        def broken(results,gates):
            decision=summarize_results(results,gates);decision['status']='pass';decision['exit_code']=0
            for gate in decision['gates']: gate['status']='pass'
            return decision
        with patch('modelaudit.scenarios.summarize_results',side_effect=broken):
            cases=build_scenarios(train,ref,predictions)
        failed={c['id'] for c in cases if not c['expectations_met']}
        self.assertTrue({'identifier_overlap','duplicate_rows','numeric_shift','sparse_segment'}<=failed)


class RawAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.shared.cleanup)
        cls.model_path=Path(cls.shared.name)/'model.joblib'
        labels=np.tile([0,1],60)
        cls.train=pd.DataFrame({'id':[f'T{i}' for i in range(120)],'x':np.arange(120)/120+(labels*2-1)*5,
                               'group':['a','a','b','b']*30,'y':labels})
        model=LogisticRegression().fit(cls.train[['x']],labels)
        joblib.dump(model,cls.model_path)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.config=self.root/'audit.yaml';self.counter=0
        self.train.to_csv(self.root/'train.csv',index=False)
        self.ref=self.train.copy();self.ref['id']=[f'R{i}' for i in range(120)]
        self.ref.to_csv(self.root/'reference.csv',index=False)
        self.ref.to_csv(self.root/'current.csv',index=False)
        shutil.copyfile(self.model_path,self.root/'model.joblib')
        self.raw={'version':2,'inputs':{'train':'train.csv','reference':'reference.csv','current':'current.csv','model':'model.joblib'},
                  'schema':{'numeric':['x'],'categorical':['group'],'target':'y','identifier':'id'},
                  'current_labeled':True,'model':{'trusted':True,'sha256':fingerprint(self.root/'model.joblib'),'features':['x']},
                  'checks':{'data':{},'evaluation':{'n_bins':1,'minimum_bin_rows':1,'thresholds':[.5]},
                            'segments':{'columns':['group']},'drift':{}},'gates':[]}

    def save(self):
        self.config.write_text(yaml.safe_dump(self.raw),encoding='utf-8')

    def run_audit(self):
        self.save();self.counter+=1
        return run_audit(self.config,self.root/f'out{self.counter}')

    def test_real_inference_and_json_roundtrip_preserve_inputs(self):
        self.save();before={p.name:fingerprint(p) for p in self.root.iterdir()}
        report=self.run_audit()
        self.assertTrue(report['model_deserialization_performed'])
        self.assertEqual(report,json.loads((self.root/'out1/report.json').read_text()))
        self.assertEqual(report['results']['evaluation']['models']['fitted']['classification']['f1'],1)
        self.assertEqual(before,{name:fingerprint(self.root/name) for name in before})

    def test_no_model_deserialization_without_explicit_trust(self):
        self.raw['model']['trusted']=False;self.save()
        with patch('modelaudit.raw_audit.joblib.load') as load,self.assertRaises(ConfigError):
            run_audit(self.config,self.root/'out')
        load.assert_not_called();self.assertFalse((self.root/'out').exists())

    def test_model_hash_mismatch_never_deserializes_or_completes(self):
        self.raw['model']['sha256']='0'*64
        with patch('modelaudit.raw_audit.joblib.load') as load,self.assertRaisesRegex(ValueError,'checksum'):
            self.run_audit()
        load.assert_not_called();self.assertTrue((self.root/'out1/failure.json').exists())
        self.assertFalse((self.root/'out1/manifest.json').exists())

    def test_missing_predictor_reports_failure_and_independent_drift(self):
        self.ref.drop(columns='x').to_csv(self.root/'reference.csv',index=False)
        report=self.run_audit()
        self.assertEqual(report['decision']['status'],'fail')
        self.assertEqual(report['results']['evaluation']['status'],'skipped')
        drift=report['results']['drift']['comparison']
        self.assertEqual(drift['features']['x']['status'],'skipped')
        self.assertEqual(drift['features']['group']['total_variation'],0)
        self.assertEqual(report['coverage']['drift']['state'],'partial')
        self.assertTrue((self.root/'out1/manifest.json').exists())

    def test_duplicate_and_blank_headers_produce_schema_findings(self):
        for header in ('id,x,group,group','id,x,group,'):
            (self.root/'reference.csv').write_text(header+'\nR0,1,a,0\nR1,2,b,1\n')
            report=self.run_audit()
            self.assertEqual(report['decision']['status'],'fail')
            self.assertTrue(report['decision']['data_failures'])

    def test_bad_numeric_value_reports_failure_without_suppressing_other_drift(self):
        current=self.ref.copy();current['x']=current.x.astype(object);current.loc[0,'x']='bad'
        current.to_csv(self.root/'current.csv',index=False)
        report=self.run_audit()
        self.assertEqual(report['decision']['status'],'fail')
        self.assertEqual(report['results']['drift']['comparison']['features']['group']['total_variation'],0)
        self.assertEqual(report['results']['drift']['comparison']['predictions']['status'],'skipped')

    def test_unlabeled_current_has_no_labeled_performance(self):
        self.raw['current_labeled']=False
        self.ref.drop(columns='y').to_csv(self.root/'current.csv',index=False)
        report=self.run_audit()
        self.assertEqual(report['results']['drift']['comparison']['performance']['status'],'skipped')
        self.assertEqual(report['results']['drift']['comparison']['predictions']['ks_distance'],0)
        self.assertFalse(any('y' in f['detail']['evidence'].get('missing_columns',[]) for f in report['decision']['data_failures']))

    def test_unavailable_gate_cannot_pass(self):
        self.raw['gates']=[{'name':'missing','path':['evaluation','not_here'],'minimum':0}]
        report=self.run_audit()
        self.assertEqual(report['decision']['gates'][0]['status'],'unavailable')
        self.assertEqual(report['decision']['exit_code'],1)

    def test_invalid_configs_do_not_create_output(self):
        original=deepcopy(self.raw)
        for change in ('bad_feature','bad_hash','different_threshold','bad_segment','bad_label','bad_schema','unknown','nonbool'):
            self.raw=deepcopy(original)
            if change=='bad_feature': self.raw['model']['features']=[[]]
            if change=='bad_hash': self.raw['model']['sha256']='bad'
            if change=='different_threshold': self.raw['checks']['segments']['threshold']=.7
            if change=='bad_segment': self.raw['checks']['segments']['columns']=['id']
            if change=='bad_label': self.raw['model']['positive_label']=True
            if change=='bad_schema': self.raw['schema']['numeric']='x'
            if change=='unknown': self.raw['checks']['drift']['bogus']=1
            if change=='nonbool': self.raw['current_labeled']='yes'
            self.save()
            with self.subTest(change=change),self.assertRaises(ConfigError):
                run_audit(self.config,self.root/'unexpected')
            self.assertFalse((self.root/'unexpected').exists())

    def test_data_only_does_not_load_model(self):
        self.raw['checks']={'data':{}};self.raw['inputs'].pop('model');self.raw.pop('model')
        with patch('modelaudit.raw_audit.joblib.load') as load:
            report=self.run_audit()
        load.assert_not_called();self.assertFalse(report['model_deserialization_performed'])

    def test_changed_input_prevents_completion(self):
        from modelaudit.raw_audit import run_raw_checks
        def changed(config):
            result=run_raw_checks(config)
            with (self.root/'reference.csv').open('a') as stream: stream.write('\n')
            return result
        with patch('modelaudit.raw_audit.run_raw_checks',side_effect=changed),self.assertRaisesRegex(ValueError,'changed'):
            self.run_audit()
        self.assertFalse((self.root/'out1/manifest.json').exists())

    def test_overwrite_and_ragged_input_are_rejected(self):
        self.run_audit()
        with self.assertRaises(FileExistsError): run_audit(self.config,self.root/'out1')
        (self.root/'reference.csv').write_text('x,y\n1,2,3\n')
        with self.assertRaises(ValueError): self.run_audit()
        self.assertFalse((self.root/'out2/manifest.json').exists())


class FullCliControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name)
        cls.first=run_cli_experiments(cls.root/'first')
        cls.second=run_cli_experiments(cls.root/'second')

    def test_both_complete_suites_match_all_expected_verdicts(self):
        for result in (self.first,self.second):
            self.assertTrue(result['all_expectations_met'])
            self.assertEqual(len(result['scenarios']),10)
            self.assertEqual(result['exit_code'],0)
            self.assertEqual(len({s['model_sha256'] for s in result['scenarios']}),2)

    def test_every_output_hash_and_repeat_measurement(self):
        for name in ('first','second'):
            root=self.root/name
            manifest=json.loads((root/'manifest.json').read_text())
            for path,digest in manifest['files'].items(): self.assertEqual(fingerprint(root/path),digest)
        for item in self.first['scenarios']:
            a=json.loads((self.root/'first'/item['report']).read_text())
            b=json.loads((self.root/'second'/item['report']).read_text())
            self.assertEqual(a['results'],b['results'])
            self.assertEqual(a['decision'],b['decision'])
        for path in (self.root/'first').rglob('*.csv'):
            self.assertEqual(path.read_bytes(),(self.root/'second'/path.relative_to(self.root/'first')).read_bytes())

    def test_assessor_rejects_missing_duplicate_wrong_and_blocked_evidence(self):
        item=next(s for s in self.first['scenarios'] if s['scenario']=='segment_degradation')
        original=json.loads((self.root/'first'/item['report']).read_text())
        for mutation in ('decision','missing','duplicate','metric','exit','gate','blocked'):
            report=deepcopy(original);exit_code=1
            if mutation=='decision': report['decision']['status']='pass'
            if mutation=='missing': report['results'].pop('segments')
            if mutation=='duplicate': report['decision']['gates'].append(deepcopy(report['decision']['gates'][0]))
            if mutation=='metric': report['results']['evaluation']['models']['fitted']['classification']['f1']=1
            if mutation=='exit': exit_code=0
            if mutation=='gate': report['decision']['gates'][1]['status']='pass'
            if mutation=='blocked': report['results']['segments']['models']={}
            with self.subTest(mutation=mutation):
                self.assertFalse(all(a['met'] for a in assess_report('segment_degradation',report,exit_code)))

    def test_scenario_mutations_are_isolated(self):
        train,reference=synthetic_frame(42,'T'),synthetic_frame(43,'R')
        before=reference.copy(deep=True)
        changed,current=scenario_inputs(train,reference,'segment_degradation')
        mask=reference.Contract=='One year'
        self.assertEqual(int(mask.sum()),100)
        pd.testing.assert_frame_equal(changed.loc[~mask],reference.loc[~mask])
        pd.testing.assert_frame_equal(changed.drop(columns='tenure'),reference.drop(columns='tenure'))
        pd.testing.assert_frame_equal(reference,before)
        self.assertIsNone(current)

    def test_existing_folder_and_bad_seed_refused(self):
        with self.assertRaises(FileExistsError): run_cli_experiments(self.root/'first')
        for seed in (-1,True,2**32-1):
            with self.assertRaises(ValueError): run_cli_experiments(self.root/'bad',seed)
        self.assertFalse((self.root/'bad').exists())

    def test_process_failure_preserves_failure_marker(self):
        with patch('modelaudit.cli_experiments.subprocess.run',return_value=subprocess.CompletedProcess([],2,'','deliberate error')):
            with self.assertRaisesRegex(ValueError,'execution failed'): run_cli_experiments(self.root/'failed')
        self.assertTrue((self.root/'failed/failure.json').exists())
        self.assertFalse((self.root/'failed/manifest.json').exists())

    def test_unexpected_verdict_returns_unmet_suite(self):
        with patch('modelaudit.cli_experiments.assess_report',return_value=[{'description':'Injected wrong verdict','met':False}]):
            # Reuse completed subprocess output to test suite aggregation without ten extra model audits.
            real_run=subprocess.run
            def previous(command,**kwargs):
                if '--output' not in command:
                    return real_run(command,**kwargs)
                target=Path(command[command.index('--output')+1])
                shutil.copytree(self.root/'first/reports'/target.name,target)
                data=json.loads((target/'report.json').read_text())
                return subprocess.CompletedProcess(command,data['decision']['exit_code'],'','')
            with patch('modelaudit.cli_experiments.subprocess.run',side_effect=previous):
                result=run_cli_experiments(self.root/'unmet')
        self.assertEqual(result['exit_code'],1)
        self.assertFalse(result['all_expectations_met'])

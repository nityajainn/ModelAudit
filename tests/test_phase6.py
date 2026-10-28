from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from modelaudit.artifacts import fingerprint
from modelaudit.audit_config import load_audit_config
from modelaudit.config import ConfigError
from modelaudit.decisions import summarize_results, evaluate_gates
from modelaudit.runner import run_audit
from modelaudit.reporting import render_reports
import test_phase3


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'audit.yaml'
        self.raw = {'version': 1, 'checks': {'drift': {'reference': 'a.csv', 'current': 'b.csv', 'numeric': ['x']}}}

    def load(self):
        self.path.write_text(yaml.safe_dump(self.raw), encoding='utf-8')
        return load_audit_config(self.path)

    def test_defaults_and_paths_resolve_relative_to_config(self):
        config = self.load()
        self.assertEqual(config['checks']['drift']['reference'], str(self.root/'a.csv'))
        self.assertEqual(config['checks']['drift']['minimum_rows'], 30)
        self.assertEqual(config['gates'], [])
        self.assertEqual(config['inputs'], {})

    def test_all_check_defaults_and_fixed_order(self):
        self.raw = {'version': 1, 'inputs': {'prepared': 'prepared', 'models': 'models'},
                    'checks': {'segments': {}, 'evaluation': {}, 'data': {}}}
        result = self.load()
        self.assertEqual(list(result['checks']), ['data', 'evaluation', 'segments'])
        self.assertEqual(result['checks']['segments']['columns'], ['Contract'])
        self.assertEqual(result['checks']['data']['missing_fraction'], .05)

    def test_defaults_are_not_shared_between_loads(self):
        first = self.load()
        first['checks']['drift']['categorical'].append('changed')
        self.assertEqual(self.load()['checks']['drift']['categorical'], [])

    def test_unknown_keys_at_every_level_fail(self):
        baseline = deepcopy(self.raw)
        for location in ('root', 'check', 'options', 'inputs', 'gate'):
            self.raw = deepcopy(baseline)
            if location == 'root': self.raw['surprise'] = 1
            elif location == 'check': self.raw['checks']['surprise'] = {}
            elif location == 'options': self.raw['checks']['drift']['surprise'] = 1
            elif location == 'inputs': self.raw['inputs'] = {'surprise': 'a'}
            else: self.raw['gates'] = [{'surprise': 1}]
            with self.subTest(location=location), self.assertRaises(ConfigError): self.load()

    def test_version_and_check_enabling_types(self):
        base = deepcopy(self.raw)
        for value in (True, '1', 2, None):
            self.raw = deepcopy(base); self.raw['version'] = value
            with self.subTest(version=value), self.assertRaises(ConfigError): self.load()
        for value in (None, False, [], {}):
            self.raw = deepcopy(base); self.raw['checks'] = value
            with self.subTest(checks=value), self.assertRaises(ConfigError): self.load()
        for value in (None, True, []):
            self.raw = deepcopy(base); self.raw['checks']['drift'] = value
            with self.assertRaises(ConfigError): self.load()

    def test_required_and_unused_inputs(self):
        for inputs in ({}, {'models': 'models'}, {'prepared': 'prepared', 'models': 'models'}):
            self.raw = {'version': 1, 'checks': {'data': {}}, 'inputs': inputs}
            with self.subTest(inputs=inputs), self.assertRaises(ConfigError): self.load()
        self.raw['inputs'] = {'prepared': 'prepared'}
        self.assertEqual(list(self.load()['inputs']), ['prepared'])

    def test_invalid_common_options_and_roles(self):
        base = deepcopy(self.raw)
        for options in ({'threshold': True}, {'threshold': float('nan')}, {'n_bins': 101},
                        {'minimum_rows': 0}, {'minimum_class_rows': 1.5}, {'maximum_categories': 1001},
                        {'numeric': 'x'}, {'categorical': ['x']}, {'probability': 'x'},
                        {'target': 'y'}, {'reference': ''}, {'current': False}):
            self.raw = deepcopy(base); self.raw['checks']['drift'].update(options)
            with self.subTest(options=options), self.assertRaises(ConfigError): self.load()

    def test_invalid_evaluation_grids(self):
        for grid in ([], [0.5, 0.5], [0.8, 0.2], [0.1, True], [1.1], ['0.1'], '0.5'):
            self.raw = {'version': 1, 'inputs': {'prepared':'p','models':'m'}, 'checks': {'evaluation': {'thresholds':grid}}}
            with self.subTest(grid=grid), self.assertRaises(ConfigError): self.load()

    def test_invalid_data_policy_and_segment_columns(self):
        for check, options in (('data', {'minimum_pairs': 1}), ('data', {'correlation_threshold':0}),
                               ('segments', {'columns':['customerID']}), ('segments', {'columns':['Contract','Contract']}),
                               ('segments', {'maximum_groups':1001})):
            self.raw = {'version':1, 'inputs': {'prepared':'p'}, 'checks':{check:options}}
            if check == 'segments': self.raw['inputs']['models']='m'
            with self.subTest(options=options), self.assertRaises(ConfigError): self.load()

    def test_gate_bounds_paths_names_and_types(self):
        valid = {'name':'distance','path':['drift','comparison','predictions','ks_distance'],'maximum':.2}
        for change in ({'name':''}, {'path':['data','counts','fail']}, {'path':['drift',True]},
                       {'path':['drift',-1]}, {'path':'drift.x'}, {'maximum':True},
                       {'maximum':float('inf')}, {'minimum':.4}, {'maximum':'0.2'}):
            self.raw['gates'] = [{**valid, **change}]
            with self.subTest(change=change), self.assertRaises(ConfigError): self.load()
        self.raw['gates']=[valid,valid]
        with self.assertRaises(ConfigError): self.load()
        self.raw['gates']=[{'name':'missing bound','path':['drift','comparison']}]
        with self.assertRaises(ConfigError): self.load()
        self.raw['gates']=[valid]
        self.assertEqual(self.load()['gates'][0]['maximum'], .2)

    def test_duplicate_keys_and_unsafe_yaml_fail(self):
        for content in ('version: 1\nversion: 1\nchecks: {}\n', '!!python/object/apply:os.system [echo unsafe]',
                        'version: 1\nchecks:\n  data:\n    minimum_pairs: 3\n    minimum_pairs: 4\n'):
            self.path.write_text(content, encoding='utf-8')
            with self.assertRaises(ConfigError): load_audit_config(self.path)


class DecisionTests(unittest.TestCase):
    def gate(self, **kwargs):
        return {'name':'criterion','path':['evaluation','metric'],'minimum':.5, **kwargs}

    def test_bounds_are_inclusive_and_can_have_both_limits(self):
        for value, expected in ((.5,'pass'), (.7,'pass'),(.49,'fail'),(.71,'fail')):
            actual = evaluate_gates({'evaluation':{'metric':value}}, [self.gate(maximum=.7)])[0]
            self.assertEqual(actual['status'], expected)

    def test_missing_undefined_boolean_nonfinite_and_container_are_unavailable(self):
        for value in (None, True, '0.8', float('nan'), float('inf'), [], {}):
            actual=evaluate_gates({'evaluation':{'metric':value}}, [self.gate()])[0]
            self.assertEqual(actual['status'],'unavailable')
            self.assertIsNone(actual['value'])
        self.assertEqual(evaluate_gates({'evaluation':{}}, [self.gate()])[0]['status'],'unavailable')

    def test_list_index_paths_and_out_of_range(self):
        results={'evaluation':{'rows':[.8]}}
        self.assertEqual(evaluate_gates(results,[self.gate(path=['evaluation','rows',0])])[0]['status'],'pass')
        self.assertEqual(evaluate_gates(results,[self.gate(path=['evaluation','rows',1])])[0]['status'],'unavailable')

    def test_warnings_skips_and_undefined_evidence_need_review(self):
        for value in ({'warnings':[{'code':'low_rows'}]}, {'status':'skipped','reason':'No labels'},
                      {'undefined_reasons':{'precision':'No predictions'}}, {'undefined_reason':'No observations'}):
            result=summarize_results({'drift':value},[])
            self.assertEqual((result['status'],result['exit_code']),('review',1))
            self.assertTrue(result['review_items'])

    def test_fail_precedes_review_and_does_not_hide_it(self):
        result=summarize_results({'data':{'findings':[{'status':'fail'},{'status':'warn'},{'status':'skipped'}]}},[])
        self.assertEqual(result['status'],'fail')
        self.assertEqual(len(result['review_items']),2)
        self.assertEqual(len(result['data_failures']),1)

    def test_no_gates_and_clean_results_pass_only_selected_checks(self):
        result=summarize_results({'evaluation':{'metric':.8}},[])
        self.assertEqual((result['status'],result['exit_code']),('pass',0))
        self.assertEqual(result['gates'],[])

    def test_gate_failure_and_unavailable_outcomes(self):
        self.assertEqual(summarize_results({'evaluation':{'metric':.1}},[self.gate()])['status'],'fail')
        self.assertEqual(summarize_results({'evaluation':{}},[self.gate()])['status'],'review')


class RunnerTests(unittest.TestCase):
    def setUp(self):
        test_phase3.PredictionArtifactTests.setUp(self)
        self.config_path=self.root/'audit.yaml'
        self.output=self.root/'run'
        self.snapshot=self.root/'snapshot.csv'
        self.snapshot.write_text('x,c,p,y\n1,A,0.1,0\n2,B,0.9,1\n3,A,0.2,0\n4,B,0.8,1\n',encoding='utf-8')
        self.drift={'reference':'snapshot.csv','current':'snapshot.csv','numeric':['x'],'categorical':['c'],
                    'probability':'p','target':'y','minimum_rows':1,'minimum_class_rows':1,'minimum_predicted_positives':1}
        self.raw={'version':1,'inputs':{'prepared':'prepared','models':'models'},
                  'checks':{'data':{},'evaluation':{},'segments':{},'drift':self.drift}}

    def save(self):
        self.config_path.write_text(yaml.safe_dump(self.raw,sort_keys=False),encoding='utf-8')

    def run_report(self):
        self.save()
        return run_audit(self.config_path,self.output)

    def drift_only(self):
        self.raw={'version':1,'checks':{'drift':self.drift}}

    def test_all_checks_match_individual_outputs_and_manifest_hashes(self):
        before={str(p):fingerprint(p) for folder in (self.prepared,self.models) for p in folder.iterdir()}
        with patch('joblib.load',side_effect=AssertionError('No deserialization')):
            result=self.run_report()
        self.assertEqual(list(result['results']),['data','evaluation','segments','drift'])
        for name,value in result['results'].items():
            self.assertEqual(json.loads((self.output/f'{name}.json').read_text()),value)
        manifest=json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(manifest['status'],'complete')
        self.assertEqual(len(manifest['files']),7)
        for name,expected in manifest['files'].items(): self.assertEqual(fingerprint(self.output/name),expected)
        self.assertEqual(json.loads((self.output/'report.json').read_text()),result)
        self.assertEqual(before,{str(p):fingerprint(p) for folder in (self.prepared,self.models) for p in folder.iterdir()})
        self.assertEqual(result['inputs']['configuration'],fingerprint(self.config_path))

    def test_selected_check_only_and_supported_identity_pass(self):
        self.drift_only()
        result=self.run_report()
        self.assertEqual(result['decision']['status'],'pass')
        self.assertEqual(list(result['results']),['drift'])
        self.assertFalse((self.output/'data.json').exists())

    def test_existing_directory_is_preserved_even_when_empty(self):
        self.output.mkdir(); sentinel=self.output/'keep.txt'; sentinel.write_text('keep')
        with self.assertRaises(FileExistsError): self.run_report()
        self.assertEqual(sentinel.read_text(),'keep')
        self.assertEqual(list(self.output.iterdir()),[sentinel])

    def test_output_cannot_be_inside_prepared_or_model_inputs(self):
        self.save()
        for folder in (self.prepared,self.models):
            with self.assertRaisesRegex(ValueError,'inside'): run_audit(self.config_path,folder/'new-run')
            self.assertFalse((folder/'new-run').exists())

    def test_invalid_config_and_missing_inputs_create_no_output(self):
        self.raw['unknown']=True
        with self.assertRaises(ConfigError): self.run_report()
        self.assertFalse(self.output.exists())
        del self.raw['unknown']; self.raw['inputs']['models']='missing'
        with self.assertRaises(ValueError): self.run_report()
        self.assertFalse(self.output.exists())

    def test_changed_config_during_load_is_rejected_before_output(self):
        self.save()
        def change(path):
            result=load_audit_config(path)
            with path.open('a') as f: f.write('\n# changed\n')
            return result
        with patch('modelaudit.runner.load_audit_config',side_effect=change):
            with self.assertRaisesRegex(ValueError,'Configuration changed'): run_audit(self.config_path,self.output)
        self.assertFalse(self.output.exists())

    def test_changed_earlier_input_during_later_check_prevents_completion(self):
        from modelaudit.drift_audit import audit_snapshot_files
        def change(**kwargs):
            result=audit_snapshot_files(**kwargs)
            with (self.prepared/'train.csv').open('a') as stream: stream.write('\n')
            return result
        with patch('modelaudit.runner.audit_snapshot_files',side_effect=change):
            with self.assertRaisesRegex(ValueError,'changed across'): self.run_report()
        self.assertFalse((self.output/'manifest.json').exists())
        self.assertEqual(json.loads((self.output/'failure.json').read_text())['status'],'failed')

    def test_failed_check_retains_partial_evidence_without_completion(self):
        with patch('modelaudit.runner.audit_baseline_predictions',side_effect=ValueError('Invalid probabilities')):
            with self.assertRaises(ValueError): self.run_report()
        self.assertTrue((self.output/'data.json').exists())
        self.assertFalse((self.output/'report.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())
        failed=json.loads((self.output/'failure.json').read_text())
        self.assertEqual(failed['completed_checks'],['data'])

    def test_report_write_failure_cannot_mark_complete(self):
        self.drift_only()
        with patch('modelaudit.runner.render_reports',side_effect=OSError('Report disk error')):
            with self.assertRaises(OSError): self.run_report()
        self.assertTrue((self.output/'drift.json').exists())
        self.assertTrue((self.output/'failure.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())

    def test_failed_gate_keeps_complete_evidence_and_nonzero_outcome(self):
        self.drift_only()
        self.raw['gates']=[{'name':'Example deliberate bound','path':['drift','comparison','predictions','ks_distance'],'minimum':.1}]
        report=self.run_report()
        self.assertEqual(report['decision']['status'],'fail')
        self.assertEqual(json.loads((self.output/'manifest.json').read_text())['exit_code'],1)

    def test_repeat_results_equal_except_runtime_context(self):
        first=self.run_report(); self.output=self.root/'repeat'; second=self.run_report()
        def clean(value):
            if isinstance(value,dict): return {k:clean(v) for k,v in value.items() if k!='context'}
            if isinstance(value,list): return [clean(v) for v in value]
            return value
        self.assertEqual(clean(first),clean(second))

    def test_summary_escapes_html_markdown_and_has_no_scripts_or_remote_assets(self):
        self.drift_only(); report=self.run_report()
        hostile='<script>alert(1)</script> ![pixel](https://example.test/pixel) | `x`'
        report['decision']['review_items']=[{'path':[hostile],'detail':hostile}]
        report['decision']['status']='review'
        md,html=render_reports(report)
        self.assertNotIn('<script>',html)
        self.assertIn('&lt;script&gt;',html)
        self.assertIn('!\\[pixel\\]',md)
        self.assertIn('&#124;',md)
        class Inspector(HTMLParser):
            def __init__(self): super().__init__(); self.tags=[]; self.links=[]
            def handle_starttag(self,tag,attrs):
                self.tags.append(tag)
                self.links.extend(v for k,v in attrs if k in ('src','href'))
        parsed=Inspector(); parsed.feed(html)
        self.assertNotIn('script',parsed.tags)
        self.assertNotIn('iframe',parsed.tags)
        self.assertEqual(parsed.links,[])
        self.assertIn('table',parsed.tags)

    def test_summary_marks_unavailable_instead_of_zero(self):
        self.drift_only(); report=self.run_report()
        report['results']['drift']['comparison']['performance']['difference']['f1']=None
        markdown,html=render_reports(report)
        self.assertIn('Unavailable',markdown)
        self.assertIn('Unavailable',html)

    def command(self,*arguments):
        return subprocess.run([sys.executable,'-m','modelaudit',*arguments],cwd=self.root,text=True,capture_output=True)

    def test_cli_help_version_and_schema_only_check(self):
        self.assertEqual(self.command('--help').returncode,0)
        self.assertIn('ModelAudit',self.command('--version').stdout)
        self.drift_only(); self.drift['reference']='not-present.csv'; self.save()
        result=self.command('check-config','--config',str(self.config_path))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('has not been audited',result.stdout)
        self.assertFalse(self.output.exists())

    def test_cli_pass_review_fail_and_error_exit_codes(self):
        self.drift_only(); self.save()
        args=['run','--config',str(self.config_path),'--output',str(self.output),'--quiet']
        passed=self.command(*args)
        self.assertEqual(passed.returncode,0,passed.stderr)
        self.assertIn('PASS',passed.stdout)
        self.assertEqual(passed.stderr,'')
        self.assertEqual(self.command(*args).returncode,2)
        self.raw['checks']['drift']['minimum_rows']=30; self.save()
        reviewed=self.command(*args,'--output',str(self.root/'review'))
        self.assertEqual(reviewed.returncode,1,reviewed.stderr)
        self.assertIn('REVIEW',reviewed.stdout)
        self.raw['gates']=[{'name':'test gate','path':['drift','comparison','predictions','ks_distance'],'minimum':.1}]; self.save()
        failed=self.command(*args,'--output',str(self.root/'failed'))
        self.assertEqual(failed.returncode,1,failed.stderr)
        self.assertIn('FAIL',failed.stdout)
        self.raw['unexpected']=True; self.save()
        invalid=self.command(*args,'--output',str(self.root/'invalid'))
        self.assertEqual(invalid.returncode,2)
        self.assertFalse((self.root/'invalid').exists())


if __name__=='__main__': unittest.main()

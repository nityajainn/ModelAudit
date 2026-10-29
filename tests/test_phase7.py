from contextlib import redirect_stdout
from copy import deepcopy
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from modelaudit.artifacts import fingerprint
from modelaudit.cli import main
from modelaudit.dataset import CATEGORICAL, IDENTIFIER, TARGET
from modelaudit.experiment_reports import render_experiments
from modelaudit.experiment_suite import _checksum_case, run_experiments
from modelaudit.model_audit import BASELINE_NAMES, _load_prediction_inputs
from modelaudit.scenarios import build_scenarios, COPY_COLUMN, NEW_CATEGORY
import test_phase3


def frame(rows, prefix, offset):
    data = {name: ['yes' if i % 2 else 'no' for i in range(rows)] for name in CATEGORICAL}
    data.update({IDENTIFIER: [f'{prefix}{i:04}' for i in range(rows)], TARGET: np.arange(rows) % 2,
                 'tenure': np.arange(rows), 'MonthlyCharges': offset + np.arange(rows)/10,
                 'TotalCharges': offset + np.arange(rows)*10})
    data['Contract'] = ['One year' if i % 3 else 'Month-to-month' for i in range(rows)]
    return pd.DataFrame(data)


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.reference = frame(80, 'T', 20), frame(40, 'R', 30)
        cls.predictions = {name: np.where(cls.reference[TARGET] == 1, high, 1-high)
                           for name, high in zip(BASELINE_NAMES, (.8, .7))}
        cls.cases = {case['id']: case for case in build_scenarios(cls.train, cls.reference, cls.predictions)}

    def test_control_preserves_existing_warnings_and_has_zero_changes(self):
        case = self.cases['control']
        self.assertTrue(case['expectations_met'])
        self.assertEqual(case['audit_outcome'], 'review')
        for comparison in case['results']['drift']['models'].values():
            self.assertEqual(comparison['predictions']['ks_distance'], 0)
            self.assertTrue(all(value == 0 for value in comparison['performance']['difference'].values()))

    def test_missingness_mutation_has_exact_count(self):
        case = self.cases['missingness']
        self.assertEqual(case['parameters']['newly_missing_rows'], 12)
        self.assertEqual(case['evidence']['finding']['evidence']['fraction'], .3)
        self.assertTrue(case['expectations_met'])

    def test_duplicates_fail_identifier_check_with_exact_count(self):
        case = self.cases['duplicate_rows']
        self.assertEqual(case['evidence']['duplicates']['evidence']['count'], 25)
        self.assertEqual(case['evidence']['identifiers']['evidence']['count'], 25)
        self.assertEqual(case['audit_outcome'], 'fail')

    def test_overlap_is_exact_and_fails(self):
        case = self.cases['identifier_overlap']
        self.assertEqual(case['evidence']['overlap']['evidence']['reference_rows'], 25)
        self.assertEqual(case['audit_outcome'], 'fail')

    def test_target_copy_detected_in_both_partitions(self):
        for finding in self.cases['target_copy']['evidence']['signals'].values():
            self.assertIn('target_copy', finding['evidence']['signals'])
            self.assertEqual(finding['status'], 'warn')

    def test_missing_column_fails_schema_without_discarding_findings(self):
        case = self.cases['missing_column']
        self.assertEqual(case['evidence']['schema']['evidence']['missing_columns'], ['Contract'])
        self.assertEqual(case['audit_outcome'], 'fail')
        self.assertGreater(case['results']['data']['counts']['skipped'], 0)

    def test_numeric_shift_measures_feature_change_without_inference(self):
        case = self.cases['numeric_shift']
        self.assertEqual(case['evidence']['MonthlyCharges']['ks_distance'], 1)
        comparison = case['results']['drift']['comparison']
        self.assertEqual(comparison['predictions']['status'], 'skipped')
        self.assertEqual(comparison['performance']['status'], 'skipped')
        self.assertEqual(case['decision']['gates'][0]['status'], 'fail')

    def test_unseen_category_is_all_rows_and_unit_distance(self):
        metric = self.cases['unseen_category']['evidence']['Contract']
        self.assertEqual(metric['total_variation'], 1)
        self.assertEqual(metric['new_category_fraction'], 1)
        self.assertEqual(metric['new_category_rows'], 40)

    def test_inversion_changes_perfect_predictions_to_wrong_rankings(self):
        case = self.cases['inverted_probabilities']
        for model in case['evidence']['models'].values():
            self.assertEqual(model['reference_roc_auc'], 1)
            self.assertEqual(model['current_roc_auc'], 0)
            self.assertEqual(model['supported_roc_auc_difference'], -1)
        self.assertEqual(case['audit_outcome'], 'fail')

    def test_permutation_hides_from_distribution_checks_but_changes_labeled_scores(self):
        case = self.cases['permuted_probabilities']
        self.assertEqual(case['parameters']['rotation_rows'], 13)
        for model in case['evidence']['models'].values():
            self.assertEqual(model['prediction_ks'], 0)
            self.assertEqual(model['prediction_histogram_tv'], 0)
            self.assertEqual(model['current_roc_auc'], 0)
        self.assertTrue(any(g['status'] == 'pass' for g in case['decision']['gates']))
        self.assertTrue(any(g['status'] == 'fail' for g in case['decision']['gates']))

    def test_sparse_cohort_withholds_comparison_and_gate(self):
        case = self.cases['sparse_segment']
        self.assertTrue(case['parameters']['selection_uses_labels'])
        for model in case['evidence']['models'].values():
            self.assertEqual(model['support']['positives'], 1)
            self.assertIsNone(model['supported_roc_auc_difference'])
        self.assertTrue(all(g['status'] == 'unavailable' for g in case['decision']['gates']))

    def test_expectation_success_is_distinct_from_audit_failure(self):
        self.assertEqual(len(self.cases), 11)
        self.assertTrue(all(case['expectations_met'] for case in self.cases.values()))
        self.assertTrue(any(case['audit_outcome'] == 'fail' for case in self.cases.values()))
        json.dumps(self.cases, allow_nan=False)

    def test_source_tables_arrays_and_repeat_measurements_are_preserved(self):
        before = self.train.copy(deep=True), self.reference.copy(deep=True)
        probabilities = {name: values.copy() for name, values in self.predictions.items()}
        second = build_scenarios(self.train, self.reference, self.predictions)
        assert_frame_equal(self.train, before[0]); assert_frame_equal(self.reference, before[1])
        for name in probabilities: np.testing.assert_array_equal(probabilities[name], self.predictions[name])
        self.assertEqual({case['id']:case for case in second}, self.cases)

    def test_invalid_control_and_reserved_category_fail(self):
        changed = self.reference.copy(); changed.loc[0, IDENTIFIER] = self.train.loc[0, IDENTIFIER]
        with self.assertRaisesRegex(ValueError, 'structural'): build_scenarios(self.train, changed, self.predictions)
        changed = self.reference.copy(); changed.loc[0, 'Contract'] = NEW_CATEGORY
        with self.assertRaisesRegex(ValueError, 'reserved'): build_scenarios(self.train, changed, self.predictions)

    def test_insufficient_observations_fail_preconditions(self):
        changed = self.reference.copy(); changed['MonthlyCharges'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Not enough observed'): build_scenarios(self.train, changed, self.predictions)
        changed = self.reference.copy(); changed['TotalCharges'] = np.nan
        with self.assertRaisesRegex(ValueError, 'observed value'): build_scenarios(self.train, changed, self.predictions)

    def test_probability_shape_type_and_range_fail_before_mutation(self):
        for array in (np.array(.5), np.ones((40,1))*.5, np.ones(39)*.5, np.full(40,2.), np.array(['.5']*40)):
            predictions = {**self.predictions, BASELINE_NAMES[0]:array}
            with self.subTest(shape=array.shape), self.assertRaises(ValueError):
                build_scenarios(self.train, self.reference, predictions)
        with self.assertRaises(ValueError): build_scenarios(self.train,self.reference,{})


class ExperimentArtifactTests(unittest.TestCase):
    def setUp(self):
        test_phase3.PredictionArtifactTests.setUp(self)
        self.output = self.root/'experiments'

    def run_suite(self):
        return run_experiments(self.prepared, self.models, self.output)

    def test_complete_suite_preserves_sources_and_hashes_every_output(self):
        before={str(p):fingerprint(p) for d in (self.prepared,self.models) for p in d.iterdir()}
        with patch('joblib.load',side_effect=AssertionError('No deserialization')):
            summary=self.run_suite()
        self.assertTrue(summary['all_expectations_met'])
        self.assertEqual(summary['exit_code'],0)
        self.assertEqual(len(summary['scenarios']),12)
        self.assertEqual(summary,json.loads((self.output/'summary.json').read_text()))
        manifest=json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(manifest['status'],'complete')
        self.assertEqual(len(manifest['files']),15)
        for name,digest in manifest['files'].items(): self.assertEqual(fingerprint(self.output/name),digest)
        self.assertEqual(before,{str(p):fingerprint(p) for d in (self.prepared,self.models) for p in d.iterdir()})
        self.assertFalse(list(self.output.rglob('*.csv')))
        self.assertFalse(list(self.output.rglob('*.npy')))

    def test_existing_output_and_protected_input_directories_are_preserved(self):
        self.output.mkdir(); (self.output/'keep.txt').write_text('keep')
        with self.assertRaises(FileExistsError): self.run_suite()
        self.assertEqual((self.output/'keep.txt').read_text(),'keep')
        for folder in (self.prepared,self.models):
            with self.assertRaises(ValueError): run_experiments(self.prepared,self.models,folder/'nested')
            self.assertFalse((folder/'nested').exists())

    def test_corrupt_actual_source_is_not_treated_as_a_successful_injection(self):
        with (self.models/'logistic_regression_probabilities.npy').open('ab') as f: f.write(b'bad source')
        with self.assertRaisesRegex(ValueError,'checksum'): self.run_suite()
        self.assertFalse(self.output.exists())

    def test_bad_bound_array_shape_is_rejected_before_output(self):
        path=self.models/'logistic_regression_probabilities.npy'; np.save(path,np.array(.5))
        self.summary['models'][BASELINE_NAMES[0]]['probabilities_sha256']=fingerprint(path)
        self.summary_path.write_text(json.dumps(self.summary))
        with self.assertRaises(ValueError): self.run_suite()
        self.assertFalse(self.output.exists())

    def test_checksum_case_accepts_only_the_specific_expected_rejection(self):
        _,_,paths,_=_load_prediction_inputs(self.prepared,self.models)
        result=_checksum_case(paths)
        self.assertEqual(result['audit_outcome'],'rejected')
        self.assertTrue(result['expectations_met'])
        self.assertFalse(result['evidence']['successful_report_written'])
        with patch('modelaudit.experiment_suite.audit_baseline_predictions',side_effect=ValueError('unrelated failure')):
            with self.assertRaisesRegex(ValueError,'unrelated'): _checksum_case(paths)

    def test_unexpected_error_retains_failed_marker_without_completion(self):
        with patch('modelaudit.experiment_suite.audit_baseline_predictions',side_effect=ValueError('unrelated failure')):
            with self.assertRaises(ValueError): self.run_suite()
        self.assertEqual(json.loads((self.output/'failure.json').read_text())['status'],'failed')
        self.assertFalse((self.output/'manifest.json').exists())

    def test_accepted_corruption_completes_as_unmet_expectation_not_success(self):
        with patch('modelaudit.experiment_suite.audit_baseline_predictions',return_value={}):
            summary=self.run_suite()
        self.assertFalse(summary['all_expectations_met'])
        self.assertEqual(summary['exit_code'],1)
        case=summary['scenarios'][-1]
        self.assertEqual(case['audit_outcome'],'accepted')
        self.assertFalse(case['expectations_met'])
        self.assertEqual(json.loads((self.output/'manifest.json').read_text())['status'],'complete')

    def test_changed_inputs_during_scenarios_stop_before_output(self):
        def change(*args):
            cases=build_scenarios(*args)
            with self.summary_path.open('a') as f: f.write('\n')
            return cases
        with patch('modelaudit.experiment_suite.build_scenarios',side_effect=change):
            with self.assertRaisesRegex(ValueError,'changed during'): self.run_suite()
        self.assertFalse(self.output.exists())

    def test_changed_inputs_during_checksum_case_prevent_completion(self):
        def change(paths):
            case=_checksum_case(paths)
            with self.summary_path.open('a') as f: f.write('\n')
            return case
        with patch('modelaudit.experiment_suite._checksum_case',side_effect=change):
            with self.assertRaisesRegex(ValueError,'changed during'): self.run_suite()
        self.assertTrue((self.output/'cases/control.json').exists())
        self.assertTrue((self.output/'failure.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())

    def test_repeat_case_bytes_and_measurements_match_except_context(self):
        first=self.run_suite(); first_dir=self.output; self.output=self.root/'repeat'; second=self.run_suite()
        first.pop('context');second.pop('context');self.assertEqual(first,second)
        for source in (first_dir/'cases').glob('*.json'):
            self.assertEqual(source.read_bytes(),(self.output/'cases'/source.name).read_bytes())

    def test_report_escaping_and_independent_success_wording(self):
        summary=self.run_suite()
        hostile='<script>alert(1)</script> ![pixel](https://example.test/pixel) | `x`'
        changed=deepcopy(summary); changed['scenarios'][0]['mutation']=hostile
        changed['scenarios'][0]['evidence']['extra']=hostile
        markdown,html=render_experiments(changed)
        self.assertNotIn('<script>',html)
        self.assertIn('&lt;script&gt;',html)
        self.assertIn('!\\[pixel\\]',markdown)
        self.assertIn('&#124;',markdown)
        self.assertIn('does not mean',html)
        class Inspector(HTMLParser):
            def __init__(self): super().__init__(); self.tags=[]; self.links=[]
            def handle_starttag(self,tag,attrs):
                self.tags.append(tag); self.links.extend(v for k,v in attrs if k in ('src','href'))
        parsed=Inspector();parsed.feed(html)
        self.assertNotIn('script',parsed.tags);self.assertEqual(parsed.links,[])

    def test_module_cli_success_overwrite_and_missing_input(self):
        args=[sys.executable,'-m','modelaudit','experiments','--prepared',str(self.prepared),'--models',str(self.models),
              '--output',str(self.output),'--quiet']
        first=subprocess.run(args,cwd=self.root,text=True,capture_output=True)
        self.assertEqual(first.returncode,0,first.stderr)
        self.assertIn('12/12 scenarios matched expectations',first.stdout)
        self.assertEqual(first.stderr,'')
        self.assertEqual(subprocess.run(args,cwd=self.root,text=True,capture_output=True).returncode,2)
        self.assertEqual(subprocess.run(args+['--models',str(self.root/'missing'),'--output',str(self.root/'other')],
                                        cwd=self.root,text=True,capture_output=True).returncode,2)
        self.assertFalse((self.root/'other').exists())

    def test_cli_unmet_expectation_returns_one(self):
        with patch('modelaudit.experiment_suite.audit_baseline_predictions',return_value={}), redirect_stdout(io.StringIO()) as stream:
            code=main(['experiments','--prepared',str(self.prepared),'--models',str(self.models),'--output',str(self.output),'--quiet'])
        self.assertEqual(code,1)
        self.assertIn('11/12 scenarios matched expectations',stream.getvalue())


if __name__=='__main__': unittest.main()

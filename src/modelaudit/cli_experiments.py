"""Synthetic end-to-end controls: fitted models, ordinary CLI, independent assertions."""

import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

from .artifacts import fingerprint, run_context, write_json
from .baselines import build_pipeline
from .dataset import FEATURES, NUMERIC, CATEGORICAL, TARGET, IDENTIFIER

SCENARIOS = ('clean','contamination','target_leakage','feature_drift','segment_degradation')


def synthetic_frame(seed, prefix):
    rng = np.random.default_rng(seed)
    labels = np.tile([0,1],500)
    frame = pd.DataFrame({IDENTIFIER:[f'{prefix}-{i:04d}' for i in range(1000)], TARGET:labels,
        'tenure':30 + (labels*2-1)*rng.uniform(5,25,1000),
        'MonthlyCharges':rng.uniform(20,120,1000), 'TotalCharges':rng.uniform(100,5000,1000)})
    for name in CATEGORICAL:
        frame[name] = np.where(rng.permutation(np.arange(1000))%2,'yes','no')
    frame['Contract'] = ['Month-to-month']*900 + ['One year']*100
    return frame.iloc[rng.permutation(1000)].reset_index(drop=True)


def scenario_inputs(train, reference, name):
    ref, current = reference.copy(deep=True), None
    if name=='clean':
        current=ref.copy(deep=True)
    elif name=='contamination':
        ref.iloc[:20] = train.iloc[:20].to_numpy()
    elif name=='target_leakage':
        ref['injected_target_copy'] = ref[TARGET].copy()
    elif name=='feature_drift':
        current=ref.drop(columns=[TARGET]).copy(deep=True)
        current['tenure'] += 100
    elif name=='segment_degradation':
        mask=ref['Contract']=='One year'
        ref.loc[mask,'tenure']=60-ref.loc[mask,'tenure']
    else:
        raise ValueError('Unknown scenario.')
    return ref,current


def assess_report(name, report, process_exit):
    """Read reported evidence; never recompute metrics or use the decision engine."""
    checks=[]
    def expect(label, met):
        checks.append({'description':label, 'met':bool(met)})
    def near(a,b):
        return a is not None and abs(a-b)<1e-9
    try:
        expected={'clean':'pass','contamination':'fail','target_leakage':'review',
                  'feature_drift':'fail','segment_degradation':'fail'}[name]
        decision=report['decision']
        expect('Expected aggregate verdict',decision['status']==expected)
        expect('Expected actual process exit',process_exit==(0 if expected=='pass' else 1))
        expect('Reported exit agrees with process',decision['exit_code']==process_exit)
        expect('Fitted inference occurred',report['model_deserialization_performed'] is True)
        results=report['results']
        findings=results['data']['findings']
        def finding(check,dataset='reference',column=None):
            found=[f for f in findings if f['check']==check and f['dataset']==dataset
                   and (column is None or f['evidence'].get('column')==column)]
            if len(found)!=1:
                raise ValueError('Missing or duplicated targeted finding: '+check)
            return found[0]
        metrics=results['evaluation']['models']['fitted']
        groups=results['segments']['models']['fitted']['segments'][0]['groups']
        yearly=[g for g in groups if g['value']=='One year']
        monthly=[g for g in groups if g['value']=='Month-to-month']
        if len(yearly)!=1 or len(monthly)!=1:
            raise ValueError('Missing or duplicated expected segment.')
        gates={g['name']:g for g in decision['gates']}
        wanted={'overall F1':'pass','annual recall':'fail' if name=='segment_degradation' else 'pass'}
        if name in ('clean','feature_drift'):
            wanted['tenure shift']='fail' if name=='feature_drift' else 'pass'
        expect('Exact gate identities and statuses',len(gates)==len(decision['gates']) and set(gates)==set(wanted)
               and all(gates[k]['status']==v for k,v in wanted.items()))
        if name=='clean':
            expect('No review items or data failures',not decision['review_items'] and not decision['data_failures'])
        if name=='contamination':
            item=finding('identifier_overlap','cross_split')
            expect('Exactly twenty contaminated identifiers fail',item['status']=='fail' and item['evidence']['reference_rows']==20)
            expect('Exactly twenty predictor overlaps',finding('predictor_overlap','cross_split')['evidence']['reference_rows']==20)
        if name=='target_leakage':
            item=finding('leakage_heuristic',column='injected_target_copy')
            expect('Audit-only exact-copy feature warns',item['status']=='warn' and 'target_copy' in item['evidence']['signals'])
            expect('Leakage feature excluded from inference','injected_target_copy' not in report['configuration']['model']['features'])
        if name=='segment_degradation':
            expect('Overall F1 hides the local defect',near(metrics['classification']['f1'],.9))
            expect('Annual recall is zero',near(yearly[0]['metrics']['recall'],0))
            expect('Other segment remains perfect',near(monthly[0]['metrics']['recall'],1))
            expect('Signed overall and best differences',near(yearly[0]['difference_from_overall']['recall'],-.9)
                   and near(yearly[0]['difference_from_best']['recall'],-1))
            expect('Exact affected-group confusion matrix',yearly[0]['confusion_matrix']==[[0,50],[50,0]])
        else:
            expect('Reference F1 remains perfect',near(metrics['classification']['f1'],1))
        if name in ('clean','feature_drift'):
            drift=results['drift']['comparison']
            expect('Every declared feature measured',set(drift['features'])==set(FEATURES)
                   and all(v.get('status')!='skipped' for v in drift['features'].values()))
            for feature,value in drift['features'].items():
                distance=value.get('ks_distance',value.get('jensen_shannon_distance'))
                expect('Expected distance: '+feature,near(distance,1 if name=='feature_drift' and feature=='tenure' else 0))
            prediction=drift['predictions']
            expect('Prediction change measured',prediction['status']=='measured' and
                   (near(prediction['ks_distance'],0) if name=='clean' else prediction['ks_distance']>.2))
            if name=='clean':
                performance=drift['performance']
                expect('Labeled identity differences',performance['status']=='measured' and
                       all(near(v,0) for v in performance['difference'].values()))
            else:
                expect('No invented current performance',drift['performance']['status']=='skipped')
    except (KeyError,TypeError,ValueError,IndexError) as exc:
        expect('Complete unambiguous expected evidence: '+str(exc),False)
    return checks


def run_cli_experiments(output, seed=42):
    if type(seed) is not int or not 0<=seed<2**32-1:
        raise ValueError('Seed must be an integer from 0 through 2**32-2.')
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    try:
        train, reference=synthetic_frame(seed,'train'),synthetic_frame(seed+1,'reference')
        train.to_csv(output/'train.csv',index=False,lineterminator='\n')
        for folder in ('models','configs','reports','cases'):
            (output/folder).mkdir()
        models={}
        for kind in ('logistic_regression','random_forest'):
            model=build_pipeline(kind,seed=seed)
            model.fit(train[FEATURES],train[TARGET])
            path=output/'models'/f'{kind}.joblib'
            with path.open('xb') as stream:
                joblib.dump(model,stream)
            models[kind]=path
        for name in SCENARIOS:
            ref,current=scenario_inputs(train,reference,name)
            case=output/'cases'/name
            case.mkdir()
            ref.to_csv(case/'reference.csv',index=False,lineterminator='\n')
            if current is not None:
                current.to_csv(case/'current.csv',index=False,lineterminator='\n')
            for kind,path in models.items():
                checks={'data':{},'evaluation':{'n_bins':1,'minimum_bin_rows':1,'thresholds':[.5]},
                        'segments':{'columns':['Contract'],'minimum_rows':50}}
                gates=[{'name':'overall F1','path':['evaluation','models','fitted','classification','f1'],'minimum':.85},
                       {'name':'annual recall','path':['segments','models','fitted','segments',0,'groups',1,'metrics','recall'],'minimum':.8}]
                inputs={'train':'../train.csv','reference':f'../cases/{name}/reference.csv','model':f'../models/{kind}.joblib'}
                if current is not None:
                    inputs['current']=f'../cases/{name}/current.csv'
                    checks['drift']={'minimum_rows':50}
                    gates.append({'name':'tenure shift','path':['drift','comparison','features','tenure','ks_distance'],'maximum':.2})
                config={'version':2,'inputs':inputs,'schema':{'numeric':NUMERIC,'categorical':CATEGORICAL,'target':TARGET,'identifier':IDENTIFIER},
                        'current_labeled':name=='clean','model':{'sha256':fingerprint(path),'trusted':True,'features':FEATURES},
                        'checks':checks,'gates':gates}
                config_path=output/'configs'/f'{name}-{kind}.yaml'
                config_path.write_text(yaml.safe_dump(config),encoding='utf-8')
                input_paths={'configuration':config_path,**{f'inputs/{key}':(config_path.parent/value).resolve() for key,value in inputs.items()}}
                before={key:fingerprint(p) for key,p in input_paths.items()}
                destination=output/'reports'/f'{name}-{kind}'
                process=subprocess.run([sys.executable,'-B','-m','modelaudit','run','--config',str(config_path),'--output',str(destination),'--quiet'],
                    cwd=output, text=True,capture_output=True,timeout=120)
                (output/'reports'/f'{name}-{kind}.log').write_text(process.stdout+process.stderr,encoding='utf-8')
                if process.returncode not in (0,1):
                    raise ValueError(f'{name}/{kind} audit execution failed: {process.stderr}')
                report=json.loads((destination/'report.json').read_text())
                manifest=json.loads((destination/'manifest.json').read_text())
                assertions=assess_report(name,report,process.returncode)
                assertions.extend([
                    {'description':'Input/config/model fingerprints agree and are unchanged','met':report['inputs']==before=={k:fingerprint(p) for k,p in input_paths.items()}},
                    {'description':'Completed audit output hashes match','met':manifest['status']=='complete' and all(fingerprint(destination/k)==v for k,v in manifest['files'].items())},
                    {'description':'Report identifies the exact configuration and model','met':report['configuration']['model']['sha256']==config['model']['sha256'] and report['configuration']['inputs']['model']==str(path)}])
                records.append({'scenario':name,'model':kind,'model_sha256':fingerprint(path),'audit_outcome':report['decision']['status'],
                    'audit_exit':process.returncode,'report':str(destination.relative_to(output)/'report.json'),
                    'expectations':assertions,'expectations_met':all(a['met'] for a in assertions)})
        result={'format_version':1,'protocol':'fitted-model-cli','seed':seed,'context':run_context(),
                'scenarios':records,'all_expectations_met':all(r['expectations_met'] for r in records),
                'exit_code':0 if all(r['expectations_met'] for r in records) else 1,
                'limitations':['Invented synthetic rows, not real customers or performance estimates.',
                    'Fixed demonstration gates are not production thresholds; review exits 1 in ModelAudit.',
                    'Two models fitted once each on clean training data; all audits reuse the same artifacts.']}
        write_json(output/'summary.json',result)
        write_json(output/'manifest.json',{'status':'complete','files':{str(p.relative_to(output)).replace('\\','/'):fingerprint(p)
            for p in sorted(output.rglob('*')) if p.is_file()}})
        return result
    except Exception as exc:
        try:
            write_json(output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'message':str(exc),'completed_audits':len(records)})
        except OSError:
            pass
        raise

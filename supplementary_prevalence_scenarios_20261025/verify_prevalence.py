from pathlib import Path
import json,math
b=Path(__file__).resolve().parent
d=json.loads((b/'prevalence_scenarios.json').read_text());inputs=json.loads((b/'prevalence_input_counts.json').read_text())
key=lambda r:tuple(r[k] for k in ['source_dataset','target_dataset','seed','model'])
lookup={key(r):r for r in inputs};assert len(lookup)==len(inputs)==180
assert len(d['rows'])==1080
seen=set()
for r in d['rows']:
 k=key(r);pi=r['assumed_attack_fraction'];assert pi in d['assumed_attack_fractions'];assert (k,pi) not in seen;seen.add((k,pi))
 q=lookup[k];tn,fp,fn,tp=[q[x] for x in ['tn','fp','fn','tp']]
 # Independent probability-weighted confusion masses from original integer counts.
 tpr=tp/(tp+fn);fpr=fp/(tn+fp);t=pi*tpr;f=(1-pi)*fpr;n=pi*(1-tpr);s=(1-pi)*(1-fpr)
 expected={'scenario_precision':t/(t+f) if t+f else None,'scenario_f1':2*t/(2*t+f+n) if 2*t+f+n else None,'scenario_accuracy':t+s,'expected_false_alerts_per_million_flows':1e6*f,'expected_true_alerts_per_million_flows':1e6*t}
 for name,value in expected.items():
  if value is None:assert r[name] is None
  else:assert math.isclose(r[name],value,abs_tol=1e-10,rel_tol=1e-12),(k,name)
assert all(sum(1 for k,pi in seen if pi==p)==180 for p in d['assumed_attack_fractions'])
print('PASS: 180 original jobs, six assumed attack fractions, 1080 scenario rows. No production measurements or new predictions.')

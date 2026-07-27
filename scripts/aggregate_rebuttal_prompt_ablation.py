#!/usr/bin/env python3
"""Aggregate four paired prompt-ablation arms into reviewer-facing evidence."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    for line in path.read_text().splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows


def paired_bootstrap_delta(a: list[int], b: list[int], seed: int=20260724, draws: int=10000) -> tuple[float,float]:
    if len(a)!=len(b) or not a: raise ValueError('paired arrays required')
    rng=random.Random(seed); n=len(a); vals=[]
    for _ in range(draws):
        idx=[rng.randrange(n) for _ in range(n)]
        vals.append(100*sum(a[i]-b[i] for i in idx)/n)
    vals.sort()
    return vals[int(0.025*draws)], vals[min(draws-1,int(0.975*draws))]


def exact_mcnemar(a: list[int], b: list[int]) -> dict[str, int | float]:
    """Return paired transitions and the two-sided exact McNemar p-value."""
    if len(a) != len(b) or not a:
        raise ValueError("paired arrays required")
    both_pass = sum(x == 1 and y == 1 for x, y in zip(a, b))
    a_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    b_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    both_fail = sum(x == 0 and y == 0 for x, y in zip(a, b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "both_pass": both_pass,
        "a_only": a_only,
        "b_only": b_only,
        "both_fail": both_fail,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def arm_key(summary: dict[str,Any]) -> str:
    spec=summary['spec']; return f"{spec['variant']}:{spec['planner']}->{spec['executor']}"


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--dataset',required=True); ap.add_argument('--output-prefix',type=Path,required=True); ap.add_argument('--variants',nargs=2,default=['original','paraphrase'],metavar=('BASELINE','ALTERNATE')); args=ap.parse_args()
    summaries=[]
    for p in sorted(args.root.glob(f"{args.dataset}_*/run_summary.json")):
        summaries.append(json.loads(p.read_text()))
    if len(summaries)!=4: raise SystemExit(f'expected 4 completed arms, found {len(summaries)}')
    hashes={s['spec']['task_manifest_sha256'] for s in summaries}; ids={tuple(s['spec']['task_ids']) for s in summaries}
    if len(hashes)!=1 or len(ids)!=1: raise SystemExit('task manifests differ across arms')
    task_ids=list(next(iter(ids))); n=len(task_ids); arms={}; sanitized=[]
    for s in summaries:
        files=list(Path(s['output_dir']).glob('output-data_*.jsonl'))
        if len(files)!=1: raise SystemExit(f"output file mismatch for {arm_key(s)}")
        rows=load_jsonl(files[0]); by={str(r['task_id']):r for r in rows}
        if set(by)!=set(task_ids): raise SystemExit(f"task coverage mismatch for {arm_key(s)}: {len(by)}/{n}")
        detail_files=list(Path(s['output_dir']).glob('detailed-results_*.jsonl'))
        if len(detail_files)!=1: raise SystemExit(f"detail file mismatch for {arm_key(s)}")
        role_totals={t:{} for t in task_ids}
        for request in load_jsonl(detail_files[0]):
            tid=str(request['task_id']); role=str(request['role'])
            bucket=role_totals.setdefault(tid,{}).setdefault(role,{'input_tokens':0,'cached_tokens':0,'output_tokens':0,'requests':0})
            bucket['input_tokens']+=int(request.get('input_tokens') or 0)
            bucket['cached_tokens']+=int(request.get('cached_tokens') or 0)
            bucket['output_tokens']+=int(request.get('output_tokens') or 0)
            bucket['requests']+=1
        scores=[1 if by[t].get('eval_passed') is True else 0 for t in task_ids]
        arms[arm_key(s)]={'scores':scores,'passed':sum(scores),'errors':sum(bool(by[t].get('errors')) for t in task_ids),'requests':sum(int(by[t].get('num_requests') or 0) for t in task_ids),'summary':s}
        spec=s['spec']
        for tid in task_ids:
            row=by[tid]
            sanitized.append({
                'dataset': args.dataset, 'task_id': tid, 'variant': spec['variant'],
                'planner': spec['planner'], 'executor': spec['executor'],
                'success': row.get('eval_passed') is True,
                'eval_score': row.get('eval_score'),
                'error_count': len(row.get('errors') or []),
                'e2e_latency_s': row.get('e2e_latency_s'),
                'tool_calls': int(row.get('tool_call_count') or 0),
                'requests': int(row.get('num_requests') or 0),
                'input_tokens': int(row.get('input_tokens') or 0),
                'output_tokens': int(row.get('output_tokens') or 0),
                'per_role': role_totals.get(tid,{}),
            })
    models=sorted({s['spec']['planner'] for s in summaries}|{s['spec']['executor'] for s in summaries})
    if len(models)!=2: raise SystemExit(f'expected two models, got {models}')
    strong='gpt-5.4' if 'gpt-5.4' in models else models[-1]; weak=next(x for x in models if x!=strong)
    baseline,alternate=args.variants
    observed_variants={s['spec']['variant'] for s in summaries}
    if observed_variants!={baseline,alternate}: raise SystemExit(f'variant mismatch: expected {[baseline,alternate]}, found {sorted(observed_variants)}')
    out={'dataset':args.dataset,'n':n,'task_manifest_sha256':next(iter(hashes)),'variants':{'baseline':baseline,'alternate':alternate},'arms':{},'direction_verdicts':{}}
    for k,v in arms.items(): out['arms'][k]={x:v[x] for x in ['passed','errors','requests']}
    for variant in [baseline,alternate]:
        sk=f'{variant}:{strong}->{weak}'; wk=f'{variant}:{weak}->{strong}'
        if sk not in arms or wk not in arms: raise SystemExit(f'missing direction arms for {variant}')
        a=arms[sk]['scores']; b=arms[wk]['scores']; delta=100*(sum(a)-sum(b))/n; lo,hi=paired_bootstrap_delta(a,b)
        out['direction_verdicts'][variant]={'strong_planner_passed':sum(a),'strong_executor_passed':sum(b),'strong_planner_minus_strong_executor_percent':delta,'paired_bootstrap_95_percent':[lo,hi],'paired_transitions':exact_mcnemar(a,b),'preferred_role_for_strong_model':'planner' if delta>0 else 'executor' if delta<0 else 'tie'}
    out['role_ranking_agreement']=out['direction_verdicts'][baseline]['preferred_role_for_strong_model']==out['direction_verdicts'][alternate]['preferred_role_for_strong_model']
    prompt_effects={}
    for planner,executor in [(strong,weak),(weak,strong)]:
        a=arms[f'{alternate}:{planner}->{executor}']['scores'];b=arms[f'{baseline}:{planner}->{executor}']['scores'];delta=100*(sum(a)-sum(b))/n;lo,hi=paired_bootstrap_delta(a,b,seed=20260725)
        effect={'comparison':f'{alternate}_minus_{baseline}','alternate_minus_baseline_percent':delta,'paired_bootstrap_95_percent':[lo,hi],'paired_transitions':exact_mcnemar(a,b)}
        if baseline=='original' and alternate=='paraphrase': effect['paraphrase_minus_original_percent']=delta
        prompt_effects[f'{planner}->{executor}']=effect
    out['prompt_effects']=prompt_effects
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    with args.output_prefix.with_name(args.output_prefix.name+'_per_task').with_suffix('.jsonl').open('w') as f:
        for row in sanitized:
            f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    lines=[f"# {args.dataset} prompt sensitivity",'',f"N={n}; manifest `{out['task_manifest_sha256']}`; failures are scored as zero.",'', '| Prompt | GPT-5.4 planner, mini executor | mini planner, GPT-5.4 executor | Preferred role for GPT-5.4 |', '|---|---:|---:|---|']
    for v in [baseline,alternate]:
        d=out['direction_verdicts'][v]; lines.append(f"| {v} | {d['strong_planner_passed']},{n} | {d['strong_executor_passed']},{n} | {d['preferred_role_for_strong_model']} |")
    lines+=['',f"Role-ranking agreement across prompts: `{out['role_ranking_agreement']}`.",'', 'Paired direction deltas:', '']
    for v,d in out['direction_verdicts'].items(): lines.append(f"- {v}: {d['strong_planner_minus_strong_executor_percent']:+.1f}% absolute; paired bootstrap 95% interval [{d['paired_bootstrap_95_percent'][0]:+.1f}%, {d['paired_bootstrap_95_percent'][1]:+.1f}%]; discordant {d['paired_transitions']['a_only']}/{d['paired_transitions']['b_only']}; exact McNemar p={d['paired_transitions']['exact_two_sided_p']:.4f}.")
    lines+=['','Prompt effects within each direction:','']
    for k,d in prompt_effects.items(): lines.append(f"- {k}: {d['alternate_minus_baseline_percent']:+.1f}% absolute; paired bootstrap 95% interval [{d['paired_bootstrap_95_percent'][0]:+.1f}%, {d['paired_bootstrap_95_percent'][1]:+.1f}%]; alternate fixed/broke {d['paired_transitions']['a_only']}/{d['paired_transitions']['b_only']}; exact McNemar p={d['paired_transitions']['exact_two_sided_p']:.4f}.")
    args.output_prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=='__main__': main()

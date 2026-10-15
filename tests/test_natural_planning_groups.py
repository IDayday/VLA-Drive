import csv
import gzip
import json

import pytest

from tools.decision_local.common import digest
from tools.decision_local.summarize_round1 import natural_input_groups, grouped_comparisons
from tools.local_interaction_mask_v2.compare_pdms import METRICS


def scored(token, log, value):
    return dict(token=token,log=log,status='ok',**{key:str(value if key=='score' else 1.) for key in METRICS})


def population(tmp_path, duplicate=False):
    scores={'a':scored('a','large_log',.95),'b':scored('b','other_log',.2)}
    score_path=tmp_path/'canonical.csv';score_path.write_text('fixed scored primary q0 bytes')
    def row(token):
        return dict(token=token,log=scores[token]['log'],anchor_index=0,plan_index=0,distribution='NATURAL',
            canonical_seed42_score=scores[token]['score'],status='KEEP' if token=='a' else 'UNCERTAIN',
            dynamic_information_common_oracle_support=token=='b',checker_road_or_kinematic=False,
            checker_known_violation=False,sampled_known_road_violation=False,sampled_known_overlap=False,
            checker_complete_4s=token=='a',strict_target_eligible=token=='a')
    rows=[row('a'),row('b'),dict(row('a'),anchor_index=1,dynamic_information_common_oracle_support=True),
          dict(row('a'),plan_index=1,distribution='SYNTHETIC',checker_road_or_kinematic=True)]
    if duplicate:rows.append(row('a'))
    with gzip.open(tmp_path/'plans.csv.gz','wt') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta=dict(schema='stored_natural_local_population_summary_v1',anchor_identity='anchor',
        canonical_scores_sha256=digest(score_path),row_file_sha256=digest(tmp_path/'plans.csv.gz'),
        scenes=2,label_identity='labels',scope='test',dynamic_definition='fixed common-support rule')
    (tmp_path/'SUMMARY.json').write_text(json.dumps(meta))
    return scores,score_path


def test_primary_natural_input_excludes_second_seed_and_recovery(tmp_path):
    scores,path=population(tmp_path)
    groups,evidence=natural_input_groups(tmp_path,scores,path,'anchor')
    assert groups['dynamic_information_common_oracle_support']=={'b'}
    assert groups['checker_road_or_kinematic']==set()
    assert groups['without_strict_target']==groups['incomplete_full_4s_support']=={'b'}
    assert groups['original_high_ge_0p9']=={'a'}
    assert evidence['anchor_identity']=='anchor'


def test_group_join_rejects_wrong_foundation_score_and_duplicate_scene(tmp_path):
    scores,path=population(tmp_path)
    with pytest.raises(ValueError,match='foundation and scored q0'):
        natural_input_groups(tmp_path,scores,path,'other_anchor')
    scores['a']['score']='.9'
    with pytest.raises(ValueError,match='score/log differs'):
        natural_input_groups(tmp_path,scores,path,'anchor')
    scores,path=population(tmp_path,duplicate=True)
    with pytest.raises(ValueError,match='exactly one primary'):
        natural_input_groups(tmp_path,scores,path,'anchor')


def test_high_input_group_keeps_new_failure_and_empty_group_is_explicit(tmp_path):
    scores,path=population(tmp_path);groups,_=natural_input_groups(tmp_path,scores,path,'anchor')
    final={'a':scored('a','large_log',0.),'b':scored('b','other_log',1.)}
    results=grouped_comparisons(final,scores,groups,50)
    high=results['original_high_ge_0p9']
    assert high['scenes']==1 and high['transitions']['nonzero_to_zero']==1
    assert high['paired']['delta_percentage_points']==pytest.approx(-95.)
    assert results['dynamic_information_common_oracle_support']['paired']['delta_percentage_points']==pytest.approx(80.)
    assert results['checker_road_or_kinematic']['status']=='EMPTY'
    assert results['checker_road_or_kinematic']['paired'] is None


def test_group_statistics_preserve_scene_weights_and_failed_scene_denominator():
    base={str(i):scored(str(i),'large' if i<2 else 'small',1.) for i in range(3)}
    first={str(i):scored(str(i),'large' if i<2 else 'small',0. if i<2 else 1.) for i in range(3)}
    first['0']['status']='failed'
    result=grouped_comparisons(first,base,{'all':set(base)},100)['all']
    assert result['scenes']==3 and result['logs']==2
    assert not result['paired']['valid']
    assert result['paired']['failed_first']==1
    assert result['paired']['delta_percentage_points']==pytest.approx(-200/3)
    assert result['transitions']['nonzero_to_zero']==2

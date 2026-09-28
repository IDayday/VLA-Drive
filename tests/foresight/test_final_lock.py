import copy
import pytest
from tools.foresight.lock_navtest import validate_models,validate_lock,SEEDS
from tools.foresight.score_pdms import identity_hash


def fixture_models():
    models=[];development={'schema':'foresight_paired_planning_v1','split':'dev','diagnostic':False,'valid':True,'complete_primary_matrix':True,'groups':{}}
    for arm in ('R','A','B','C','D'):
        identity={'identity':arm,'scope':'formal','updates':100000,'source_sha':'shared-source','data':{'identity':'train'},'ego':{'identity':'ego'},
                  'selected_index_hash':'full-train-index','scene_count':101592,'global_batch':32,'world_size':4,'micro_batch':8,
                  'schedule':{'horizon':100000},'precision':'bf16/fp32','config':{'framework':{'action_model':{'num_inference_timesteps':10}}}}
        checkpoint={'arm':arm,'training_seed':42,'scope':'formal','completed':100000,'run_identity':arm,'sha256':'model-'+arm}
        status={'identity':arm,'status':'COMPLETE','completed':100000,'exposure':3199952}
        models.append((identity,checkpoint,status))
        development['groups'][arm+'_train42']={'valid':True,'sampling_seeds':SEEDS,'checkpoint':copy.deepcopy(checkpoint)}
    return models,development


def test_final_lock_requires_whole_completed_matched_matrix_and_actual_dev_checkpoint():
    models,report=fixture_models();records,steps,contract=validate_models(models,report)
    assert len(records)==5 and steps==10 and contract['updates']==100000
    for change in ('startup','paused','earlier','different_exposure','different_source','wrong_development','missing_seed'):
        cases=copy.deepcopy(models);dev=copy.deepcopy(report)
        if change=='startup':cases[2][0]['scope']='startup'
        elif change=='paused':cases[2][2]['status']='PAUSED'
        elif change=='earlier':cases[2][1]['completed']=90000
        elif change=='different_exposure':cases[2][2]['exposure']+=1
        elif change=='different_source':cases[2][0]['source_sha']='other'
        elif change=='wrong_development':dev['groups']['B_train42']['checkpoint']['sha256']='other'
        else:dev['groups']['B_train42']['sampling_seeds']=[42]
        with pytest.raises(ValueError):validate_models(cases,dev)
    with pytest.raises(ValueError):validate_models(models[:-1],report)


def test_final_export_rejects_modified_lock_model_source_population_and_noise_protocol():
    checkpoint={'sha256':'model','arm':'D'};current={'identity':'current','index_sha256':'index'}
    lock={'schema':'foresight_navtest_lock_v1','checkpoints':['model'],'checkpoint_records':{'model':checkpoint},
          'evaluation_source_sha':'source','current_data_identity':'current','current_index_identity':'index',
          'scene_count':12146,'sampling_seeds':SEEDS,'inference_steps':10}
    lock['identity']=identity_hash(lock)
    validate_lock(lock,checkpoint,'source',current,12146,42,10)
    bad=copy.deepcopy(lock);bad['inference_steps']=20
    with pytest.raises(ValueError):validate_lock(bad,checkpoint,'source',current,12146,42,10)
    for args in ((dict(checkpoint,arm='C'),'source',current,12146,42,10),
                 (checkpoint,'changed',current,12146,42,10),(checkpoint,'source',dict(current,index_sha256='new'),12146,42,10),
                 (checkpoint,'source',current,12145,42,10),(checkpoint,'source',current,12146,99,10),(checkpoint,'source',current,12146,42,20)):
        with pytest.raises(ValueError):validate_lock(lock,*args)

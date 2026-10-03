import json
import pytest

from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
from tools.action_video_foresight.reconcile_development import complete_export,validate_results


def test_recovery_requires_complete_registered_fp32_current_export(tmp_path):
    data=tmp_path/'data';data.mkdir();bank=tmp_path/'predictions';bank.mkdir()
    index=[{'token':'a','log':'one'},{'token':'b','log':'two'}]
    (data/'index.json').write_text(json.dumps(index))
    reg_root=tmp_path/'registrations';reg_root.mkdir();reg_path=reg_root/'run.json'
    reg_path.write_text(json.dumps({'arm':'S4','run_id':'run','training_source_sha':'source'}))
    training={'candidate':{'name':'C1'},'registration_sha256':file_sha256(reg_path)}
    training['identity']=identity_hash(training)
    run_root=tmp_path/'students'/'run';run_root.mkdir(parents=True)
    (run_root/'identity.json').write_text(json.dumps(training))
    plan={'campaign_root':str(tmp_path),'dev_data':str(data),'dev_scenes':2,'training_source_sha':'source',
        'runs':{'S4':{'run_id':'run','gpus':[0,1]}}}
    identity={'checkpoint':{'arm':'C1','run_identity':training['identity'],'completed':5000,'scope':'formal','training_source_sha':'source'},
        'current_identity':{'split':'dev','index_sha256':identity_hash(index)},'limit':0,'world_size':2,
        'protocol':{'precision':'FP32','tf32':False,'future_conditioning':False,'scorer':None,
            'candidates_per_scene':1,'steps':10,'auxiliary_heads_removed':True}}
    (bank/'identity.json').write_text(json.dumps(identity))
    assert not complete_export(plan,'S4',5000,bank)
    for shard in range(2):
        (bank/f'shard_{shard}.json').write_text(json.dumps({'identity_sha256':identity_hash(identity),'status':'complete','completed':1}))
    assert complete_export(plan,'S4',5000,bank)
    identity['checkpoint']['run_identity']='different-model'
    (bank/'identity.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='experimental arm'):complete_export(plan,'S4',5000,bank)
    identity['checkpoint']['run_identity']=training['identity']
    identity['protocol']['future_conditioning']=True
    (bank/'identity.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='deployment protocol'):complete_export(plan,'S4',5000,bank)


def test_recovered_results_retain_full_population_and_canonical_runtime():
    plan={'dev_scenes':2};runtime={'runtime_versions':{'scipy':'fixed'}}
    score={'valid':True,'failed':0,'scenes':2,'evaluator_identity':runtime,'PDMS':.8,'metrics':{'NC':1.}}
    ego={'valid':True,'failed':0,'scenes':2,'groups':{'all':{'ADE':.5,'FDE':1.,'yaw_MAE_rad':.1}}}
    validate_results(plan,score,ego,runtime)
    with pytest.raises(ValueError,match='runtime'):validate_results(plan,score,ego,{'runtime_versions':{}})
    score['failed']=1
    with pytest.raises(ValueError,match='Incomplete'):validate_results(plan,score,ego,runtime)

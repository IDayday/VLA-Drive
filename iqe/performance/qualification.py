"""A continuation certificate binds measured evidence, without changing Base identity."""
from pathlib import Path
from ..io import read_json,file_hash
from ..contracts import require


def validate_continuation(path,options,implementation_hashes):
    certificate=read_json(path)
    require(certificate.get('status')=='PASS','runtime has no passed continuation certificate')
    require(certificate['options']==options.identity(),'qualification runtime options mismatch')
    require(certificate['implementation_hashes']==implementation_hashes,'qualification implementation changed')
    evidence={}
    for kind,item in certificate['evidence'].items():
        require(file_hash(item['path'])==item['checksum'],'qualification evidence checksum: '+kind)
        evidence[kind]=read_json(item['path'])
    require({'throughput','numerics','resume'}<=evidence.keys(),'missing runtime evidence')
    require(evidence['throughput']['slowest_rank_per_step']['n']>=100 and not evidence['throughput']['trace_run_not_throughput'], 'insufficient steady-state throughput evidence')
    require(evidence['throughput']['identity']['options']==options.identity(),'throughput used another runtime')
    require(evidence['numerics']['sampled_numerical_criteria_passed'],'numerical comparison failed')
    require(evidence['resume']['status']=='PASS','resume comparison failed')
    for kind in ('numerics','resume'):
        folder=Path(evidence[kind]['candidate'] if kind=='numerics' else evidence[kind]['resumed'])
        require(read_json(folder/'LAUNCH.json')['identity']['options']==options.identity(),kind+' evidence used another runtime')
    return certificate

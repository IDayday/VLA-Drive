"""Lead-approved one-off bounded data.pkl schema read; no unpickling or storage."""
from pathlib import Path
import collections
import hashlib
import json
import pickletools
import time
import zipfile

TARGET=Path('/mnt/project/st-world-policy-20261016/formal/consequence/runs/C1_rollout_std_fulltrain_seed42_official_4096_v1/checkpoints/step_000000_1792350704120235575/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt')
class Counted:
    def __init__(self,raw,size):self.raw=raw;self.size=size;self.reads=[];self.total=0
    def read(self,n=-1):
        at=self.raw.tell()
        if n<0:n=self.size-at
        if n<0 or self.total+n>2*2**20:raise ValueError('Metadata byte request exceeds2MiB hard cap')
        value=self.raw.read(n);self.reads.append((at,len(value)));self.total+=len(value);return value
    def seek(self,*args):return self.raw.seek(*args)
    def tell(self):return self.raw.tell()
    def seekable(self):return True
    def readable(self):return True


def main():
    assert TARGET.stat().st_size==819516336
    started=time.monotonic()
    with TARGET.open('rb',buffering=0) as raw:
        stream=Counted(raw,TARGET.stat().st_size)
        with zipfile.ZipFile(stream) as archive:
            entries=archive.infolist();matches=[item for item in entries if item.filename.endswith('/data.pkl')]
            assert len(matches)==1
            info=matches[0];assert info.file_size<=2**20 and info.compress_size<=2**20
            payload=archive.read(info);assert len(payload)==info.file_size
            storage=[item for item in entries if '/data/' in item.filename]
            ordered=sorted(entries,key=lambda item:item.header_offset)
            for item in storage:
                next_at=next((other.header_offset for other in ordered if other.header_offset>item.header_offset),archive.start_dir)
                assert not any(at<next_at and at+size>item.header_offset for at,size in stream.reads),'Unexpected tensor storage-region read'
        ops=list(pickletools.genops(payload))
        globals_=collections.Counter(arg for op,arg,pos in ops if op.name=='GLOBAL')
        strings=collections.Counter(arg for op,arg,pos in ops if op.name in ('BINUNICODE','SHORT_BINUNICODE','UNICODE'))
        assert 'deepspeed.utils.tensor_fragment fragment_address' in globals_
        assert strings['param_slice_mappings'] and strings['numel'] and strings['start']
        samples={}
        for index,(op,arg,pos) in enumerate(ops):
            if op.name in ('BINUNICODE','SHORT_BINUNICODE','UNICODE') and arg in ('numel','start'):
                samples[arg]={'literal_position':pos,'following_ops':[(a.name,b) for a,b,c in ops[index+1:index+5]]}
        result={'schema':'real_native_data_pkl_schema_only_v1','actual_asset_path':str(TARGET),'actual_asset_bytes_stat':TARGET.stat().st_size,
            'asset_full_hash_computed':False,'native_tensor_loaded':False,'torch_load_or_pickle_unpickle_used':False,
            'zip_record':info.filename,'data_pkl_bytes':len(payload),'data_pkl_sha256':hashlib.sha256(payload).hexdigest(),
            'zip_total_bytes_read_including_directory':stream.total,'read_intervals':stream.reads,
            'tensor_storage_records':len(storage),'tensor_storage_region_bytes_read':0,'GLOBAL_symbols':dict(globals_),
            'important_literal_counts':{name:strings[name] for name in ('param_slice_mappings','numel','start')},
            'first_literal_field_opcode_witnesses':samples,
            'preliminary_reader_guard_attempt':'Earlier attempt rejected read(-1) before any bytes were read; corrected to bound remaining bytes from current metadata offset.',
            'scope':'Nonexecuting pickletools schema inventory only, not full checkpoint checksum or value restoration proof',
            'wall_seconds':time.monotonic()-started,'GPU_calls':0,'official_score_calls':0}
        out=Path(__file__).with_name('REAL_NATIVE_PICKLE_SCHEMA.json');out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))

if __name__=='__main__':main()

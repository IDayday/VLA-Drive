"""Copy trusted processed metadata into a task-owned version; do not change source caches."""
import argparse
import json
import pickle
from pathlib import Path


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        return super().find_class(module.replace('numpy._core','numpy.core'),name)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifests',nargs='+',required=True);p.add_argument('--source',required=True)
    p.add_argument('--sensor-roots',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);dest=out/'meta/train';dest.mkdir(parents=True,exist_ok=False)
    tokens=sorted(set(t for m in a.manifests for t in json.loads(Path(m).read_text())))
    failures=[]
    for token in tokens:
        try:
            with (Path(a.source)/(token+'.pkl')).open('rb') as stream:raw=NumpyCompatUnpickler(stream).load()
            for cam in ['cam_f0','cam_l0','cam_r0']:
                old=Path(raw['glo_images'][cam]['image_paths'][3]);relative=Path(*old.parts[-3:])
                candidates=[Path(r)/relative for r in a.sensor_roots]
                found=next((p for p in candidates if p.is_file()),None)
                if found is None:raise FileNotFoundError(str(relative))
                raw['glo_images'][cam]['image_paths'][3]=str(found)
            with (dest/(token+'.pkl')).open('wb') as stream:pickle.dump(raw,stream,protocol=4)
        except Exception as error:failures.append({'token':token,'error':repr(error)})
    report={'requested':len(tokens),'failures':failures,'source':a.source,'sensor_roots':a.sensor_roots}
    (out/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    if failures:raise RuntimeError('Preparation failed; no samples may be silently excluded')

if __name__=='__main__':main()

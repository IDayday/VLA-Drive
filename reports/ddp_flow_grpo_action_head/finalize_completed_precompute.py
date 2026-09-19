import json, pathlib, subprocess, sys, time
root=pathlib.Path(sys.argv[1]); producer=root/'runs/action_head/precompute_rl2_control/result.json'
deadline=time.monotonic()+21600
while not producer.is_file():
    if time.monotonic()>deadline: raise TimeoutError('own precompute producer exceeded six-hour deadline')
    time.sleep(15)
result=json.loads(producer.read_text())
if result.get('status')!='PASS' or result.get('exit_codes')!=[0]:
    raise RuntimeError('own precompute producer failed: '+str(result))
subprocess.run([sys.executable,'-m','scripts.analysis.precompute_action_features','--config','configs/flow_grpo/frozen_action_head_epoch1.yaml','--finalize'],check=True)

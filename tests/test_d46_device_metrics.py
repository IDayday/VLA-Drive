import torch
from starVLA.rl.d46_fpo_engineering.device_metrics import DeviceMetrics


def test_bulk_transfer_preserves_native_scalar_values_and_sum_order():
    rows=[{'reference_KL_joint':torch.tensor(.1,dtype=torch.float64),'loss':torch.tensor(x)}for x in(.3,-.2,.6)]
    expected={};kl=0.
    for r in rows:
        for k,v in r.items():expected[k]=expected.get(k,0)+float(v)/len(rows)
        kl+=float(r['reference_KL_joint'])
    meter=DeviceMetrics()
    for r in rows:meter.add(r)
    actual,new_kl=meter.drain();assert actual==expected and new_kl==kl and not meter.rows

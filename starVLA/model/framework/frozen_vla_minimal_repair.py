"""Camera-only minimal repair, reusing the verified one-S0 deployment path."""
from torch import nn
from .frozen_vla_consequence_refiner import FrozenVLAConsequenceRefiner
from starVLA.model.modules.decision_local.minimal_repair_contract import CurrentFeatures,PlanInputs,MinimalRepairRefiner


class _ActualPlanAdapter(nn.Module):
    def __init__(self,refiner):
        super().__init__()
        if not isinstance(refiner,MinimalRepairRefiner):raise TypeError('Versioned repair refiner required')
        self.refiner=refiner

    def camera_only(self,inputs,queries,predicted=None):
        current=CurrentFeatures(inputs.H_A,inputs.W,inputs.state,inputs.body)
        plan=PlanInputs(inputs.q0,inputs.q0_encoded,queries.times)
        return self.refiner.camera_only(current,plan,queries,predicted)


class FrozenVLAMinimalRepairRefiner(FrozenVLAConsequenceRefiner):
    def __init__(self,anchor,refiner,predictor=None,*,mode='R_MIN_DIFF',**kwargs):
        modes={'LOG_NONE':'E_NONE','MIN_NONE':'E_NONE','R_MIN_CURRENT':'R_CURRENT_PRED',
               'R_MIN_ABS':'R_FUTURE_ABS','R_MIN_DIFF':'R_FUTURE_DIFF',
               'N_NONE':'E_NONE','R_CURRENT':'R_CURRENT_PRED',
               'R_ABS_FUTURE':'R_FUTURE_ABS','R_DIFF_FUTURE':'R_FUTURE_DIFF'}
        if mode not in modes:raise ValueError('Oracle repair arms cannot enter camera-only deployment')
        super().__init__(anchor,_ActualPlanAdapter(refiner),predictor,mode=modes[mode],**kwargs)
        self.repair_mode=mode

    def predict(self,observation,*,sampling_seed=42):
        # Base implementation strictly rejects GT/solver/status fields and calls
        # native S0 encode_current and its original ten-step sampler exactly once.
        result=super().predict(observation,sampling_seed=sampling_seed)
        result['plan_in']=result['q0']
        result['repair_mode']=self.repair_mode
        return result

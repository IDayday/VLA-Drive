import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.agents.EpisodeDrive.layers.world_model.ema_register_target import EMARegisterTargetCallback


class UpdateAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(()))
        self._initialized=True
        self.updates=[]
    def name(self):
        return 'numerical_hook_fixture'
    def _uses_planreg_optimizer_groups(self):
        return True
    def forward(self,features):
        return self.weight*features['x']
    def compute_loss(self,features,targets,prediction):
        return {'loss':prediction.square().mean()}
    def get_optimizers(self,total_optimizer_steps):
        return [torch.optim.AdamW(self.parameters(),lr=.001)]
    def update_ema_after_optimizer_step(self,optimizer_step,total_optimizer_steps):
        self.updates.append(optimizer_step)


def test_actual_lightning_accumulation_updates_once_and_observes_moments(tmp_path):
    agent=UpdateAgent()
    model=AgentLightningModule(agent,diagnostics={'precision_log_interval':1})
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=1,max_steps=2,
                       accumulate_grad_batches=2,logger=False,enable_checkpointing=False,
                       enable_progress_bar=False,enable_model_summary=False,
                       default_root_dir=str(tmp_path),callbacks=[EMARegisterTargetCallback()])
    loader=DataLoader([({'x':torch.ones(1)},{}) for _ in range(4)],batch_size=1)
    trainer.fit(model,loader)
    assert agent.updates==[1,2]
    assert agent.weight.item()!=1.
    import json
    audit=json.loads((tmp_path/'run_metadata/PRECISION_CONTRACT.json').read_text())
    assert audit['moments_observed'] and set(audit['adam_moment_storage'])=={'torch.float32'}


def test_optimizer_not_executed_does_not_update_ema(monkeypatch):
    from types import SimpleNamespace
    agent=UpdateAgent()
    model=AgentLightningModule(agent)
    model._trainer=SimpleNamespace(global_step=0,estimated_stepping_batches=2)
    optimizer=torch.optim.AdamW(agent.parameters())
    monkeypatch.setattr(pl.LightningModule,'optimizer_step',lambda *args,**kwargs:None)
    model.optimizer_step(0,0,optimizer,lambda:None)
    assert agent.updates==[]

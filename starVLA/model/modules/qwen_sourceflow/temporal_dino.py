"""Full nine-image Qwen evidence and current-only, training-only C1 supervision."""
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from .model import QwenSourceFlowVLA
from .qwen_encoder import QwenObservationProcessor, CAMERAS


class TemporalQwenProcessor(QwenObservationProcessor):
    """One official prefill; real chronological images, no image replication."""
    def __call__(self, observations):
        messages, images, views, frame_indices = [], [], [], []
        for obs in observations:
            state = obs['proprio'];frames = obs['frames']
            if len(state)!=8 or len(frames)!=3 or [f['offset'] for f in frames]!=[-2,-1,0]:
                raise ValueError('Three chronological camera frames and current legal state required')
            relative = [float(f['relative_seconds']) for f in frames]
            if not relative[0]<relative[1]<relative[2]==0.:
                raise ValueError('Historical timestamps must be strictly before current')
            content = [dict(type='text', text=(
                'Plan the ego vehicle trajectory for the next 4 seconds. '
                f'Navigation: {obs["navigation"]}. '
                f'Current velocity (m/s): x={float(state[0]):.4f}, y={float(state[1]):.4f}. '
                f'Current acceleration (m/s^2): x={float(state[2]):.4f}, y={float(state[3]):.4f}. '
                'Two historical frames and the current frame follow chronologically. '
                'Each frame contains synchronized left/front/right camera views.'))]
            cameras, times = [], []
            for frame in frames:
                content.append(dict(type='text', text=f'Frame relative to current (seconds): {float(frame["relative_seconds"]):.6f}.'))
                present = 0
                for camera in CAMERAS:
                    image = frame['images'].get(camera)
                    content.append(dict(type='text', text=f'Camera {camera}: '+('missing.' if image is None else '')))
                    if image is None:continue
                    content.append(dict(type='image', image=image));images.append(image)
                    cameras.append(CAMERAS.index(camera));times.append(frame['offset']);present+=1
                if frame['offset']==0 and present==0:
                    raise ValueError('No current camera is available; infrastructure/data error')
            messages.append([dict(role='user', content=content)])
            views.append(cameras);frame_indices.append(times)
        text=[self.processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
        batch=self.processor(text=text,images=images,padding=True,return_tensors='pt')
        batch['camera_indices']=views;batch['image_time_indices']=frame_indices
        return batch


class CurrentDINOHead(nn.Module):
    """Per-patch affine readout; all full-token evidence still goes to the actor.

    The historical C1 targets are 6x8 per view,1024 channels, last post-norm
    DINOv3-L/16 patches, mean-pooled2x2. The different Qwen merger grid requires
    explicit full-field bilinear resampling; this is an interface adaptation.
    """
    def __init__(self, hidden=2048, channels=1024, grid=(6,8)):
        super().__init__();self.projection=nn.Linear(hidden,channels);self.grid=tuple(grid);self._resize_weights={}

    def resize(self,patches):
        # align_corners=False bilinear interpolation expressed as two fixed
        # linear maps. CUDA atomic interpolation backward is avoided, retaining
        # the native deterministic-algorithms numerical contract.
        h,w=patches.shape[-2:];key=(h,w,patches.device)
        if key not in self._resize_weights:
            weights=[]
            for n,m in zip((h,w),self.grid):
                coordinate=((torch.arange(m,dtype=torch.float64)+.5)*n/m-.5).clamp(0,n-1)
                lo=coordinate.floor().long();hi=(lo+1).clamp_max(n-1);fraction=coordinate-lo
                matrix=torch.zeros(m,n,dtype=torch.float64)
                matrix.scatter_add_(1,lo[:,None],(1-fraction)[:,None]);matrix.scatter_add_(1,hi[:,None],fraction[:,None])
                weights.append(matrix.float().to(patches.device))
            self._resize_weights[key]=weights
        wy,wx=self._resize_weights[key]
        with torch.autocast(patches.device.type,enabled=False):
            return torch.matmul(torch.matmul(wy,patches.float()),wx.T)

    def forward(self, encoded):
        hidden=encoded['final_hidden'];predictions=[];valid=[]
        for b,layout in enumerate(encoded['image_layout']):
            by_camera={i['camera']:i for i in layout if i['frame_time']==0}
            views=[];mask=[]
            for camera in range(3):
                item=by_camera.get(camera)
                if item is None:
                    # A graph-connected zero retains consistent DDP reduction.
                    views.append(self.projection(hidden[b,:1]).sum()*0.+hidden.new_zeros((self.projection.out_features,*self.grid)))
                    mask.append(False);continue
                start=item['offset'];h,w=item['height'],item['width']
                indices=encoded['visual_indices'][b,start:start+h*w]
                patches=hidden[b,indices].T.reshape(1,hidden.shape[-1],h,w)
                resized=self.resize(patches)
                views.append(self.projection(resized[0].permute(1,2,0)).permute(2,0,1));mask.append(True)
            predictions.append(torch.stack(views));valid.append(mask)
        return torch.stack(predictions),torch.tensor(valid,device=hidden.device,dtype=torch.bool)


def current_dino_loss(prediction,target,valid,view_valid):
    if prediction.shape!=target.shape or valid.shape!=(len(prediction),3,*prediction.shape[-2:]):
        raise ValueError('C1 current teacher shape mismatch')
    if valid.dtype!=torch.bool or view_valid.dtype!=torch.bool:raise ValueError('Explicit boolean masks required')
    mask=(valid & view_valid[:,:,None,None])[:,:,None].expand_as(prediction)
    if not mask.flatten(1).any(1).all():raise ValueError('Every training scene needs a valid current teacher view')
    if not torch.isfinite(target[mask]).all() or not torch.isfinite(prediction[mask]).all():
        raise FloatingPointError('Nonfinite valid current DINO target/prediction')
    error=(torch.where(mask,prediction.float(),0.)-torch.where(mask,target.detach().float(),0.)).square()
    # Preserve the previous C1 masked_regression global-element normalization.
    # DDP averages gradients, so compensate by world size after global count.
    count=mask.sum().detach().double();world=dist.get_world_size() if dist.is_initialized() else 1
    if world>1:dist.all_reduce(count)
    return error.sum()*world/count.to(error.dtype)


class TemporalDINOFlowVLA(QwenSourceFlowVLA):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.provider.history_frames=2
        self.current_dino_head=CurrentDINOHead(self.qwen.hidden_dim)

    def forward(self,observation_batch,proprio,normalized_action,*,dino_target,dino_valid,head_only=False,**fm_randomness):
        encoded=self.qwen(observation_batch)
        if head_only:
            fm=normalized_action.new_zeros(());terms={}
        else:
            condition=self.provider(encoded,proprio)
            fm,terms=self.policy(condition,normalized_action,**fm_randomness)
        prediction,view_valid=self.current_dino_head(encoded)
        dino=current_dino_loss(prediction,dino_target,dino_valid,view_valid)
        return fm+dino,dict(**terms,FM=fm.detach(),current_DINO=dino.detach())

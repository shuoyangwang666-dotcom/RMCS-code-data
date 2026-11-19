from pathlib import Path
"""Single-observation CPU API for V5; physical storage shared by angle and spectrum."""
from functools import lru_cache
import json,time
import numpy as np
import torch
from .data import angle_image
from .mobile_networks import TinyAngleCNN
from .compact_physics import sketched_rgb
from .shared_physics import SharedPhysics

class SharedRuntime:
    def __init__(self,root,method):
        self.root,self.method=root,method;self.policy=json.loads((root/'policy.json').read_text(encoding='utf-8'))
        self.angles=np.asarray(self.policy['angles']);self.selector=None;self.net=None;self.weight=None
        if method=='traditional':
            self.bank=[SharedPhysics.load(root/'models/physics_final'/f'angle_{a}.npz') for a in range(41)]
            self.offsets=np.r_[0,np.cumsum([m.q.shape[1] for m in self.bank])]
            self.mask=np.concatenate([m.mask for m in self.bank]);self.packed=np.concatenate([m.q for m in self.bank],axis=1)
            for a,m in enumerate(self.bank):m.q=self.packed[:,self.offsets[a]:self.offsets[a+1]]
        else:
            self.net=TinyAngleCNN(41)
            self.net.load_state_dict(torch.load(root/'models/tiny_angle_final.pt',map_location='cpu',weights_only=True)['state_dict']);self.net.eval()
        self.get=lru_cache(maxsize=1)(lambda a:SharedPhysics.load(root/'models/physics_final'/f'angle_{a}.npz'))
        if method=='pure_ml':
            source=json.loads((root/'models/spectral_source.json').read_text(encoding='utf-8'))
            with np.load((self.root/'models'/source['path']) if not Path(source['path']).is_absolute() else Path(source['path'])) as f:
                self.side=int(f['side']);self.weight=torch.from_numpy(f['weight'].astype(np.float32)/float(f['input_divisor']))

    @torch.inference_mode()
    def predict(self,rgb,exposure,profile=False):
        marks={};last=time.perf_counter()
        def stamp(name):
            nonlocal last
            now=time.perf_counter();marks[name]=(now-last)*1000;last=now
        scale=30000./exposure/255.;confidence=1.;info={};z=None
        if self.method=='traditional':
            x=sketched_rgb(rgb,6144,scale);stamp('preprocessing')
            coeff=x@self.packed;energies=np.add.reduceat(coeff**2*self.mask,self.offsets[:-1]);a=int(energies.argmax())
            model=self.bank[a];z=coeff[self.offsets[a]:self.offsets[a+1]].astype(np.float64)/model.scale;stamp('angle_and_projection')
        else:
            g=torch.from_numpy(angle_image(rgb,32)[None,None]);stamp('preprocessing')
            p=self.net(g).softmax(1)[0].numpy();a=int(p.argmax());confidence=float(p.max());stamp('angle')
            if self.method=='pure_ml':
                xx=torch.from_numpy(np.asarray(rgb,np.float32).transpose(2,0,1)[None]*scale)
                values=torch.nn.functional.adaptive_avg_pool2d(xx,(self.side,self.side)).flatten(1)
                raw=torch.nn.functional.linear(values,self.weight)[0].numpy();stamp('spectrum')
                return float(self.angles[a]),raw,{'confidence':confidence,'profile_ms':marks}
            x=sketched_rgb(rgb,6144,scale);stamp('sensor_embedding');model=self.get(a);stamp('angle_cache')
            z=model.project(x);stamp('projection')
        initial=model.solve(z);stamp('initial_svd');action=self.policy['fixed_action']
        outside=float(np.sqrt(max(0.,float(x@x)-float(z@z)*model.scale**2)/max(float(x@x),1e-20)))
        called=self.method=='hybrid' and outside>self.policy['noise_gate_threshold']
        if called:
            if self.selector is None:
                import lightgbm as lgb
                self.selector=lgb.Booster(model_str=(self.root/'models/selector.txt').read_text(encoding='utf-8'))
            p=self.selector.predict(model.features(x,initial,confidence)[None],num_threads=1)[0];choice=int(p.argmax())
            if p[choice]>=self.policy['selector_threshold']:action=choice
        stamp('selector');solver,gamma=self.policy['actions'][action]
        raw=initial if solver=='svd' else model.solve(z,solver,gamma);stamp('selected_solver')
        info.update(confidence=confidence,solver=solver,gamma=gamma,selector_evaluated=called,
            residual=model.residual(x,raw,z),closed_loop_triggered=False,closed_loop_accepted=False)
        stamp('residual');info['profile_ms']=marks
        # Benchmark API records the same trigger. The archive pipeline executes
        # BM3D/UM once when needed; callers must not mistake this for feedback time.
        info['feedback_required']=self.method=='hybrid' and info['residual']>.35 and confidence<.55
        return float(self.angles[a]),raw,info

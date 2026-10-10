import argparse, json, os
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--video')
AppLauncher.add_app_launcher_args(p); a=p.parse_args(); a.enable_cameras=bool(a.video)
app=AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import tancho_v3_lab.tasks
cfg=parse_env_cfg('TanchoV3-Squat-Play-v0',device=a.device,num_envs=1)
cfg.episode_length_s=25
cfg.commands.climb.auto_prob=0; cfg.commands.climb.random_press_prob=0; cfg.commands.climb.burst_prob=0
cfg.commands.base_velocity.ranges.lin_vel_x=(0,0); cfg.commands.base_velocity.ranges.ang_vel_z=(0,0)
cfg.commands.base_velocity.resampling_time_range=(1e6,1e6)
env=gym.make('TanchoV3-Squat-Play-v0',cfg=cfg,render_mode='rgb_array' if a.video else None)
if a.video: env=gym.wrappers.RecordVideo(env,video_folder=a.video,step_trigger=lambda k:k==0,video_length=790,name_prefix='squat',disable_logger=True)
core=env.unwrapped; robot=core.scene['robot']; term=core.command_manager.get_term('climb')
ids,_=robot.find_joints(['joint_thigh_L','joint_calf_L','joint_thigh_R','joint_calf_R'],preserve_order=True)
wids,_=robot.find_bodies(['wheel_L','wheel_R'],preserve_order=True)
policy=torch.jit.load(a.checkpoint,map_location=core.device).eval()
obs,_=env.reset(); z0=float(robot.data.root_pos_w[0,2]); wheel0=robot.data.body_pos_w[0,wids,2].clone()
contact=core.scene['contact_forces']; cids,_=contact.find_bodies(['wheel_L','wheel_R'],preserve_order=True)
missing=[0,0]; longest=[0,0]; rows=[]; failed=None
with torch.inference_mode():
 for k in range(800):
  t=k*core.step_dt
  if k in (100,300,500): term.random_mode[0]=torch.tensor([1.,1.],device=core.device); term.random_timer[0]=.2
  obs,_,done,_,_=env.step(policy(obs['policy']))
  q=(robot.data.joint_pos[0,ids]-robot.data.default_joint_pos[0,ids]).tolist()
  g=robot.data.projected_gravity_b[0]; tilt=float(torch.acos((-g[2]).clamp(-1,1))*180/torch.pi)
  rows.append(dict(t=t+.02,drop=z0-float(robot.data.root_pos_w[0,2]),q=q,tilt=tilt,wheel_rise=float((robot.data.body_pos_w[0,wids,2]-wheel0).max())))
  if t>=1:
   force=contact.data.net_forces_w[0,cids].norm(dim=-1).tolist()
   for j in range(2):
    missing[j]=missing[j]+1 if force[j]<1 else 0
    longest[j]=max(longest[j],missing[j])
  if bool(done[0]): failed=t+.02; break
cycles=[]
for start in (2,6,10):
 crouch=[r for r in rows if start+.45<=r['t']<=start+.9]
 returned=[r for r in rows if start+1.6<=r['t']<=start+2.5]
 cycles.append(dict(start_s=start,mean_drop_m=sum(r['drop'] for r in crouch)/len(crouch) if crouch else None,max_return_error_rad=max(max(abs(x) for x in r['q']) for r in returned) if returned else None))
summary=dict(failed_at_s=failed,max_tilt_deg=max(r['tilt'] for r in rows),max_wheel_rise_m=max(r['wheel_rise'] for r in rows),cycles=cycles)
summary['longest_wheel_contact_below_1N_s']=[round(k*.02,3) for k in longest]
summary['pass_all']=failed is None and summary['max_tilt_deg']<10 and summary['max_wheel_rise_m']<.01 and all(c['mean_drop_m'] is not None and c['mean_drop_m']>.015 and c['max_return_error_rad'] is not None and c['max_return_error_rad']<.1 for c in cycles)
Path(a.output).write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
env.close(); os._exit(0)

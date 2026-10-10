import argparse, json, os
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--video'); p.add_argument('--task',default='TanchoV3-DeepSquat-Play-v0'); p.add_argument('--hold-durations',type=float,nargs='+',default=[1.0,3.0,8.0])
AppLauncher.add_app_launcher_args(p); a=p.parse_args(); a.enable_cameras=bool(a.video)
app=AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import tancho_v3_lab.tasks
cfg=parse_env_cfg(a.task,device=a.device,num_envs=1)
starts=[]; cursor=2.0
for duration in a.hold_durations:
 starts.append(cursor); cursor+=duration+3.0
cfg.episode_length_s=max(25.0,cursor+1.0)
cfg.commands.climb.auto_prob=0; cfg.commands.climb.random_press_prob=0; cfg.commands.climb.burst_prob=0
cfg.commands.base_velocity.ranges.lin_vel_x=(0,0); cfg.commands.base_velocity.ranges.ang_vel_z=(0,0)
cfg.commands.base_velocity.resampling_time_range=(1e6,1e6)
env=gym.make(a.task,cfg=cfg,render_mode='rgb_array' if a.video else None)
if a.video: env=gym.wrappers.RecordVideo(env,video_folder=a.video,step_trigger=lambda k:k==0,video_length=int(cfg.episode_length_s/.02),name_prefix='squat-hold',disable_logger=True)
core=env.unwrapped; robot=core.scene['robot']; term=core.command_manager.get_term('climb')
ids,_=robot.find_joints(['joint_thigh_L','joint_calf_L','joint_thigh_R','joint_calf_R'],preserve_order=True)
wids,_=robot.find_bodies(['wheel_L','wheel_R'],preserve_order=True)
policy=torch.jit.load(a.checkpoint,map_location=core.device).eval()
obs,_=env.reset(); z0=float(robot.data.root_pos_w[0,2]); wheel0=robot.data.body_pos_w[0,wids,2].clone()
contact=core.scene['contact_forces']; cids,_=contact.find_bodies(['wheel_L','wheel_R'],preserve_order=True)
missing=[0,0]; longest=[0,0]; rows=[]; failed=None
total_steps=int(cfg.episode_length_s/core.step_dt)
with torch.inference_mode():
 for k in range(total_steps):
  t=k*core.step_dt
  holding=any(start<=t<start+duration for start,duration in zip(starts,a.hold_durations))
  if holding:
   term.random_mode[0]=torch.tensor([1.,1.],device=core.device); term.random_timer[0]=1.5*core.step_dt
  obs,_,done,_,_=env.step(policy(obs['policy']))
  q=(robot.data.joint_pos[0,ids]-robot.data.default_joint_pos[0,ids]).tolist()
  g=robot.data.projected_gravity_b[0]; tilt=float(torch.acos((-g[2]).clamp(-1,1))*180/torch.pi)
  w,x,y,z=robot.data.root_quat_w[0].tolist()
  rzx,rzy,rzz=2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)
  chassis_bottom=float(robot.data.root_pos_w[0,2])-.09*abs(rzx)-.06*abs(rzy)-.045608*abs(rzz)-.000093*rzz
  rows.append(dict(t=t+.02,drop=z0-float(robot.data.root_pos_w[0,2]),chassis_bottom=chassis_bottom,q=q,tilt=tilt,wheel_rise=float((robot.data.body_pos_w[0,wids,2]-wheel0).max())))
  if t>=1:
   force=contact.data.net_forces_w[0,cids].norm(dim=-1).tolist()
   for j in range(2):
    missing[j]=missing[j]+1 if force[j]<1 else 0
    longest[j]=max(longest[j],missing[j])
  if bool(done[0]): failed=t+.02; break
cycles=[]
for start,duration in zip(starts,a.hold_durations):
 crouch=[r for r in rows if start+.8<=r['t']<=start+duration]
 released_at=start+duration
 returned=[r for r in rows if released_at+.8<=r['t']<=released_at+2.0]
 heights=[r['chassis_bottom'] for r in crouch]
 target=(-.36,.89,-.36,.89)
 cycles.append(dict(start_s=start,hold_duration_s=duration,mean_drop_m=sum(r['drop'] for r in crouch)/len(crouch) if crouch else None,mean_chassis_bottom_height_m=sum(heights)/len(heights) if heights else None,hold_height_range_m=max(heights)-min(heights) if heights else None,max_hold_joint_error_rad=max(max(abs(x-y) for x,y in zip(r['q'],target)) for r in crouch) if crouch else None,max_hold_tilt_deg=max(r['tilt'] for r in crouch) if crouch else None,max_return_error_rad=max(max(abs(x) for x in r['q']) for r in returned) if returned else None))
summary=dict(failed_at_s=failed,max_tilt_deg=max(r['tilt'] for r in rows),max_wheel_rise_m=max(r['wheel_rise'] for r in rows),cycles=cycles)
summary['longest_wheel_contact_below_1N_s']=[round(k*.02,3) for k in longest]
summary['height_reference']='lowest corner of base_link collision box above flat ground; target 0.15m +/-0.015m'
summary['pass_all']=failed is None and summary['max_tilt_deg']<10 and max(longest)*.02<=.04 and all(c['mean_chassis_bottom_height_m'] is not None and abs(c['mean_chassis_bottom_height_m']-.15)<.015 and c['hold_height_range_m'] is not None and c['hold_height_range_m']<.02 and c['max_hold_joint_error_rad'] is not None and c['max_hold_joint_error_rad']<.12 and c['max_return_error_rad'] is not None and c['max_return_error_rad']<.1 for c in cycles)
Path(a.output).write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
env.close(); os._exit(0)

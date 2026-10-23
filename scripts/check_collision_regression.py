"""Check the cylinder/box contact that previously appeared 275 m away.

Only two geoms, no robot, policy or solver. The wider cylinder comes from the
20261023_212106 failure, not the current 45 mm wheel model. Frozen float32
transforms preserve rounding from states_step_049217. A translated copy tests
the same geometry near the origin. Run on the training GPU with --device cuda:0.
"""
import argparse
from importlib.metadata import version
import json

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


def check_collision(device='cpu'):
  model=mujoco.MjModel.from_xml_string('''<mujoco>
    <option ccd_tolerance="1e-6" ccd_iterations="50"/>
    <worldbody><geom name="box" type="box" size="4 4 .5"/>
      <body><freejoint/><geom name="wheel" type="cylinder"
        size=".086 .025907457" margin=".001" condim="3" mass="1"/>
      </body></worldbody></mujoco>''')
  positions=np.array([[36,-20,-.5],[31.987443923950195,-18.52556800842285,.08856675028800964]],dtype=np.float32)
  rotations=np.array([np.eye(3),[
    [.19469423592090607,-.9806028604507446,.022640973329544067],
    [-.1358521282672882,-.004098281264305115,.990720808506012],
    [-.9714106321334839,-.19596344232559204,-.13401485979557037],
  ]],dtype=np.float32)
  positions=np.stack([positions,positions-np.array([32,-20,0],dtype=np.float32)])
  wp.init()
  with wp.ScopedDevice(device):
    wm=mjw.put_model(model)
    data=mjw.make_data(model,nworld=2,nconmax=64,njmax=256)
    mjw.kinematics(wm,data)
    wp.synchronize()
    data.geom_xpos.assign(positions)
    data.geom_xmat.assign(np.broadcast_to(rotations,(2,2,3,3)).copy())
    mjw.collision(wm,data)
    wp.synchronize()
    count=int(data.nacon.numpy()[0])
    worlds=data.contact.worldid.numpy()[:count]
    points=data.contact.pos.numpy()[:count]
    normals=data.contact.frame.numpy()[:count]
    distances=data.contact.dist.numpy()[:count]
  if not all(np.isfinite(a).all() for a in (points,normals,distances)):
    raise RuntimeError('Non-finite cylinder/box contact')
  radius=float(np.linalg.norm([.086,.025907457]))
  distances_to_wheel=np.linalg.norm(points-positions[worlds,1],axis=1)
  if np.any(distances_to_wheel>radius+.01):
    raise RuntimeError(f'Impossible contact: distance to wheel {distances_to_wheel.tolist()} m')
  if np.any(np.abs(points-positions[worlds,0])>np.array([4,4,.5])+.01):
    raise RuntimeError('Contact outside the terrain box')
  if not all(np.any(worlds==i) for i in range(2)):
    raise RuntimeError('Expected near-touching contact missing in a frozen pose')
  return dict(device=device,versions={n:version(n) for n in ('mujoco','mujoco-warp','warp-lang')},
              contacts=count,max_distance_to_wheel_m=float(distances_to_wheel.max()))


if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device',default='cpu')
  print(json.dumps(check_collision(parser.parse_args().device),indent=2))

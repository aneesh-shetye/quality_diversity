def agent_v14(state) -> tuple[float, float, float]:
  """
  The input to this function is the current state of gym's hopper v-5 environment. 
  The state has 2 vectors qpos and qvel  

  qpos is a 5 dimensional vector where:  
  qpos[0] gives the z-coordinate of the torso (height of the hopper,
  qpos[1] gives the angle of the torso, 
  qpos[2] gives the angle of the thigh joint, 
  qpos[3] gives the angle of the foot joint and, 
  qpos[4] gives the velocity of the x-coordinate (height) of the torso

  qvel is a 6 dimensional vector where: 
  qvel[0] gives the velocity of the x-coordinate of the torso 
  qvel[1] gives the velocity of the z-coordinate of the torso 
  qvel[2] gives the angular velocity of the angle of the torso 
  qvel[3] gives the angular velocity of the thigh hinge 
  qvel [4] gives the angular velocity of the leg hinge 
  qvel[5] gives the angular velocity of the foot hinge 

  """
  return (0.9, 0.0, 0.0) 
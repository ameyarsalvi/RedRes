# Characterizing the impact of actuation redundancies in skid-steer visual servoing with DRL


Porject Motivation : 
To investigate if DRL realizes emergent behaviors to compensate for
(a) parametric uncertainties modeled as uncertainties in state transitions
(b) equivalent policies transfered to reality with parametric uncertainties
when constrained to
(i) sensor throttle
(ii) variable tracking velocity




Project Goal:
Characterize realized trajectory from the metrics of (i)velcoity tracking, and, (ii)success rate, when subject to:
(a) bsln
(b) bsln + sensor dropout
(c) bsln + varying velocities



## Date : Jan 28, 2025 (Baseline)

Baseline policy trained
1. Policy trained only for tracking velocity with following Reward :

` self.rew =(1-norm_act0_eff)**2 + (1-norm_act1_eff)**2 + 10*(1 - norm_err_vel)**2 `

where : 

```
self.track_vel = -0.4*np.sin(self.velCoeff*0.05*self.step_no) + 0.5
self.velCoeff = random.uniform(0.314,0.157)
```
The velcoeff ensurs the peak for the sine vel tracking occurs at t =5s to t=10s

2. The policy is termed as : 2WE



## Date : Jan 29, 2025 (Baseline + Positional Uncertainty)

1. Policy trained tracking velocity with following Reward :

` self.rew =(1-norm_act0_eff)**2 + (1-norm_act1_eff)**2 + 10*(1 - norm_err_vel)**2 `

where : 

```
self.track_vel = -0.4*np.sin(self.velCoeff*0.05*self.step_no) + 0.5
self.velCoeff = random.uniform(0.314,0.157)
```
The velcoeff ensurs the peak for the sine vel tracking occurs at t =5s to t=10s

2. New training introduces uncertainity in state transitions modeled as :

```
self.Xerr = 0.01*random.uniform(0, 1)
self.Yerr = 0.01*random.uniform(0, 1)
```

The uncertainty is introduced for `point = np.array([self.current_pose[0]+ self.Xerr, self.current_pose[1]+ self.Yerr])` to verify in noisy pose is inside bounds

3. Policy saved as : 2WsUn
(use 8.5 mil checkpoint for evaluations)

## Date : Jan 30, 2025 (Baseline + Positionaly Uncertainty + Sensor Dropout)

1. Policy trained tracking velocity with following Reward :

` self.rew =(1-norm_act0_eff)**2 + (1-norm_act1_eff)**2 + 10*(1 - norm_err_vel)**2 `

where : 

```
self.track_vel = -0.4*np.sin(self.velCoeff*0.05*self.step_no) + 0.5
self.velCoeff = random.uniform(0.314,0.157)
```
The velcoeff ensurs the peak for the sine vel tracking occurs at t =5s to t=10s

2. New training introduces uncertainity in state transitions modeled as :

```
self.Xerr = 0.01*random.uniform(0, 1)
self.Yerr = 0.01*random.uniform(0, 1)
```

The uncertainty is introduced for `point = np.array([self.current_pose[0]+ self.Xerr, self.current_pose[1]+ self.Yerr])` to verify in noisy pose is inside bounds

3. Updated training now introduces sensor dropout for image sensor modeled as :
```
self.throttle = 10*random.randint(1, 10) 
```
Varying the refresh rate between 0.5s to 5s

4. Policy saved as : 2WsUnTr

## Date : Feb 1, 2025 (Baseline + Positionaly Uncertainty + Sensor Dropout) > Camera Motion

1. Policy trained tracking velocity with following Reward :

` self.rew =(1-norm_act0_eff)**2 + (1-norm_act1_eff)**2 + (1-norm_act2_eff)**2 + 10*(1 - norm_err_vel)**2 `

where : 

```
self.track_vel = -0.4*np.sin(self.velCoeff*0.05*self.step_no) + 0.5
self.velCoeff = random.uniform(0.314,0.157)
```
The velcoeff ensurs the peak for the sine vel tracking occurs at t =5s to t=10s

2. New training introduces uncertainity in state transitions modeled as :

```
self.Xerr = 0.01*random.uniform(0, 1)
self.Yerr = 0.01*random.uniform(0, 1)
```

The uncertainty is introduced for `point = np.array([self.current_pose[0]+ self.Xerr, self.current_pose[1]+ self.Yerr])` to verify in noisy pose is inside bounds

3. Updated training now introduces sensor dropout for image sensor modeled as :
```
self.throttle = 10*random.randint(1, 10) 
```
Varying the refresh rate between 0.5s to 5s

4. Camera motion is introduce as an action as `cam_ang = 45*action[2]` . The New environment file is also has additional C : huskyCP_gymRRC

5. Policy saved as : 2WsUnTrCb

Note : This variant seems unsuccessful but shows increasing reward trends when the training terminated.
Plans to retrain three variants bootstrapping off the policies at 8.5mil steps.

## Date : Feb 8, 2025 (Baseline + Positionaly Uncertainty + Sensor Dropout) > Camera Motion (Continued Training)

1. Policy trained tracking velocity with following Reward :

` self.rew =(1-norm_act0_eff)**2 + (1-norm_act1_eff)**2 + (1-norm_act2_eff)**2 + 10*(1 - norm_err_vel)**2 `

where : 

```
self.track_vel = -0.4*np.sin(self.velCoeff*0.05*self.step_no) + 0.5
self.velCoeff = random.uniform(0.314,0.157)
```
The velcoeff ensurs the peak for the sine vel tracking occurs at t =5s to t=10s

2. New training introduces uncertainity in state transitions modeled as :

```
self.Xerr = 0.01*random.uniform(0, 1)
self.Yerr = 0.01*random.uniform(0, 1)
```

The uncertainty is introduced for `point = np.array([self.current_pose[0]+ self.Xerr, self.current_pose[1]+ self.Yerr])` to verify in noisy pose is inside bounds

3. Updated training now introduces sensor dropout for image sensor modeled as :
```
self.throttle = 10*random.randint(1, 10) 
```
Varying the refresh rate between 0.5s to 5s

4. Camera motion is introduce as an action as `cam_ang = 45*action[2]` . The New environment file is also has additional C : huskyCP_gymRRC

5. Policy trained for 11.5 mil more steps and thus a total of 2e7 steps. Policy still uncessful. The policy freezes the camera angle to -45 Degress and tries to move the robot. 

6. Policy saved as : 2WsUnTrCb

Note : New attempt will try to address the above problem by elimination of effort penalty for camera action. This will ensure exactly same rewards for all the policies.

## Date : Feb 9, 2025 (Baseline + Positionaly Uncertainty + Sensor Dropout) -- updated reward

1. Policy trained tracking velocity with following Reward. Obsereve reward is different from prior with getting rid of penalty for camera action effort :

` self.rew =(1-norm_act0_eff)**2 + (1-norm_act1_eff)**2 + 10*(1 - norm_err_vel)**2 `

where : 

```
self.track_vel = -0.4*np.sin(self.velCoeff*0.05*self.step_no) + 0.5
self.velCoeff = random.uniform(0.314,0.157)
```
The velcoeff ensurs the peak for the sine vel tracking occurs at t =5s to t=10s

2. New training introduces uncertainity in state transitions modeled as :

```
self.Xerr = 0.01*random.uniform(0, 1)
self.Yerr = 0.01*random.uniform(0, 1)
```

The uncertainty is introduced for `point = np.array([self.current_pose[0]+ self.Xerr, self.current_pose[1]+ self.Yerr])` to verify in noisy pose is inside bounds

3. Updated training now introduces sensor dropout for image sensor modeled as :
```
self.throttle = 10*random.randint(1, 10) 
```
Varying the refresh rate between 0.5s to 5s

4. Camera motion is introduce as an action as `cam_ang = 45*action[2]` . The New environment file is also has additional C : huskyCP_gymRRC

5. Policy trained for 11.5 mil more steps and thus a total of 2e7 steps. Policy still uncessful. The policy freezes the camera angle to -45 Degress and tries to move the robot. 

6. Policy saved as : 2WsUnTrCc

Note : This was a complete failure with the agent barely learning anything. The next trial will be to avoid throttle to see if having camera motion makes a difference at all.
Image for reference
![image](https://github.com/user-attachments/assets/411bc9f7-965d-40ff-97d7-a15cb7006ccb)




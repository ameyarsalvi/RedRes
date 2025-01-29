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



## Date : Jan 28, 2024

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



## Date : Jan 29, 2024

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


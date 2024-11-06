#!/usr/bin/env bash

cd ~/CoppeliaSim_Edu_V4_8_0_rev0_Ubuntu22_04/ \
&& ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 -GwsRemoteApi.port=23053 /home/asalvi/code_workspace/Husky_CS_SB3/PoseEnhancedVN/PolicyMix_VS.ttt


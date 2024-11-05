#!/usr/bin/env bash

cd ~/CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu22_04/

for i in {4..12..2}
do
	konsole --noclose --new-tab -e ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$((23000+i)) -GwsRemoteApi.port=$((23050+1+i)) ~/code_workspace/Husky_CS_SB3/HuskyModels/PolicyMix/PolicyMix_VS.ttt && /bin/bash &
	sleep 5
done

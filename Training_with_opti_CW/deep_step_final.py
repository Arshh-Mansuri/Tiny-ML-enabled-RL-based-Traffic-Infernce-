import os
import sys
import optparse
import subprocess
import random
import numpy as np
import time
from Dqn import Learner
import matplotlib.pyplot as plt

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
'''try:
    sys.path.append("/home/gustavo/Downloads/sumo-0.32.0/tools")
    from sumolib import checkBinary
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')") '''      
PORT = 8873
import traci

def makemap(TLIds):
    maptlactions = []
    n_phases = list_of_n_phases(TLIds)
    for n_phase in n_phases:
        mapTemp = []
        if len(maptlactions) == 0:
            for i in range(n_phase):
                if i%2 == 0:
                   maptlactions.append([i])
        else:
            for state in maptlactions:
                for i in range(n_phase):
                    if i%2 == 0:
                        mapTemp.append(state+[i])
            maptlactions = mapTemp
    return maptlactions

def list_of_n_phases(TLIds):
    n_phases = []
    for light in TLIds:
        print("1: " + traci.trafficlight.getRedYellowGreenState(light))
        n_phases.append(int((len(traci.trafficlight.getRedYellowGreenState(light)) ** 0.5) * 2))
    return n_phases


def get_state(detectorIDs):
    state = []
    for detector in detectorIDs:
        speed = traci.inductionloop.getLastStepMeanSpeed(detector)
        state.append(speed)
    for detector in detectorIDs:
        veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
        state.append(veh_num)
    state = np.array(state)
    state = state.reshape((1, state.shape[0]))
    return state


def calc_reward(state, next_state):
    rew = 0
    lstate = list(state)[0]
    lnext_state = list(next_state)[0]
    for ind, (det_old, det_new) in enumerate(zip(lstate, lnext_state)):
        if ind < len(lstate)/2:
            rew += 1000*(det_new - det_old)
        else:
            rew += 1000*(det_old - det_new)

    return rew


def main():
    # Control code here
    sumoBinary = checkBinary('sumo-gui')
    sumoProcess = subprocess.Popen([sumoBinary, "-c", "/home/arsh/BTEP/DITLACS_Git/CodeBase/Map/cross.sumocfg", "--tripinfo-output",
                                    "tripinfo.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)#Map/cross.sumocfg

    traci.init(PORT)
    TLIds = traci.trafficlight.getIDList()
    actionsMap = makemap(TLIds)
    detectorIDs = traci.inductionloop.getIDList()
    state_space_size = traci.inductionloop.getIDCount()*2
    action_space_size = len(actionsMap)
    agent = Learner(state_space_size, action_space_size, 0.0) 
    state = get_state(detectorIDs)                    
    total_reward = 0
    simulationSteps = 0
    #Making the lists to store the data for plots
    waiting_times = []
    CO2_emissions=[]
    fuel_consumps=[]
    queue_lengths=[] 
    average_speeds=[] 
    #Through put
    
    while simulationSteps < 100:
        for i in range(2):
            traci.simulationStep()
            time.sleep(0.4)         
        simulationSteps += 2
        next_state = get_state(detectorIDs)
        reward = calc_reward(state, next_state)
        total_reward += reward
        sum = 0
        print(total_reward + random.randint(0,9))
        state = next_state
        waiting_time_1 = traci.edge.getLastStepHaltingNumber('-133305531#1')
        waiting_time_2 = traci.edge.getLastStepHaltingNumber('-133305558#3')
        waiting_time_3 = traci.edge.getLastStepHaltingNumber('-133305531#2')
        waiting_time_4 = traci.edge.getLastStepHaltingNumber('-133305558#2')
        waiting_time_avg=(waiting_time_1+waiting_time_2+waiting_time_3+waiting_time_4)/4
        waiting_times.append(waiting_time_avg)   
        ### 
        CO2_emission_1 = traci.edge.getCO2Emission('-133305531#1')
        CO2_emission_2 = traci.edge.getCO2Emission('-133305558#3')
        CO2_emission_3 = traci.edge.getCO2Emission('-133305531#2')
        CO2_emission_4 = traci.edge.getCO2Emission('-133305558#2')
        CO2_avg=(CO2_emission_1+CO2_emission_2+CO2_emission_3+CO2_emission_4)/4
        CO2_emissions.append(CO2_avg)  
        ###
       # getFuelConsumption(self, edgeID)
        # 
        fuel_consump_1 = traci.edge.getFuelConsumption('-133305531#1')
        fuel_consump_2 = traci.edge.getFuelConsumption('-133305558#3')
        fuel_consump_3 = traci.edge.getFuelConsumption('-133305531#2')
        fuel_consump_4 = traci.edge.getFuelConsumption('-133305558#2')
        fuel_consump_avg=(fuel_consump_1+fuel_consump_2+fuel_consump_3+fuel_consump_4)/4
        fuel_consumps.append(fuel_consump_avg) 
        
        # getLastStepLength(self, edgeID)
        # 
        queue_length_1 = traci.edge.getLastStepLength('-133305531#1')
        queue_length_2 = traci.edge.getLastStepLength('-133305558#3')
        queue_length_3= traci.edge.getLastStepLength('-133305531#2')
        queue_length_4 = traci.edge.getLastStepLength('-133305558#2')
        queue_length_avg=(queue_length_1 +queue_length_2+queue_length_3+queue_length_4 )/4
        queue_lengths.append(queue_length_avg)  
  	#getLastStepMeanSpeed(self, edgeID)
        mean_speed_1=traci.edge.getLastStepMeanSpeed('-133305531#1')
        mean_speed_2=traci.edge.getLastStepMeanSpeed('-133305558#3')
        mean_speed_3= traci.edge.getLastStepMeanSpeed('-133305531#2')
        mean_speed_4= traci.edge.getLastStepMeanSpeed('-133305558#2')
        mean_speed_avg=(mean_speed_1+mean_speed_2+mean_speed_3+mean_speed_4)/4
        average_speeds.append(mean_speed_avg)

    traci.close()
    agent.save_keras_model("keras_model")
    agent.convert_to_tflite("keras_model")

    # Plot waiting time vs episodes
    episodes = list(range(1, len(waiting_times) + 1))

    # Plot Waiting Time vs Episodes
    plt.figure()  # Create a new figure
    plt.plot(episodes, waiting_times, marker='o')
    plt.xlabel('Episodes')
    plt.ylabel('Total Waiting Time')
    plt.title('Waiting Time vs Episodes')
    #plt.show()  # Display the figure
    plt.savefig('waiting_time_vs_episodes.png')  # Save the figure

    # Plot CO2 Emissions vs Episodes
    plt.figure()  # Create a new figure
    plt.plot(episodes, CO2_emissions, marker='o', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('CO2 Emissions')
    plt.title('CO2 Emissions vs Episodes')
    #plt.show()  # Display the figure
    plt.savefig('CO2_emissions_vs_episodes.png')  # Save the figure


    # Plot Fuel consumption vs Episodes
    plt.figure()  # Create a new figure
    plt.plot(episodes, fuel_consumps, marker='o', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Fuel consumption')
    plt.title('Fuel consumption vs Episodes')
    #plt.show()  # Display the figure
    plt.savefig('fuel_consumption_vs_episodes.png')  # Save the figure


    #   Plot Queue Length vs Episodes
    plt.figure()  # Create a new figure
    plt.plot(episodes, queue_lengths, marker='o', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Queue Length')
    plt.title('Queue Length vs Episodes')
    #plt.show()  # Display the figure
    plt.savefig('queue_length_vs_episodes.png')  # Save the figure
    
    #   Plot Average Speed vs Episodes
    plt.figure()  # Create a new figure
    plt.plot(episodes, average_speeds, marker='o', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Average Speed')
    plt.title('Average Speed vs Episodes')
    #plt.show()  # Display the figure
    plt.savefig('Average_Speed_vs_episodes.png')  # Save the figure

        


    
if __name__ == '__main__':
    main()

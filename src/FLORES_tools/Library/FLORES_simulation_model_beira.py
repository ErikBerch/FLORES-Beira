"""
This file is the main simulation of the MODOS-model. The MODOS-model can be used to compare various flood risk reduction
strategies and consists of a Simulation part and an Evaluation part. the Simulation requires four .py files:

MODOS_simulation_model_<Project>_<Version>.py
    simulation_definitions_<Project>_<Version>.py
    simulation_data_<Project>_<Version>.py
    simulation_calculations_<Project>_<Version>.py

This file simulates the impact of a flood event (due to storm surge and/or rain) on a vulnerable coastal region. Here,
it is possible to implement a combination of measures to reduce the risk of flooding. The performance of a flood risk
reduction strategy is measures in the ability to reduce risk, construction costs and other case-related (non-)economic
performance metrics.

Project: Beira, Mozambique
Last updated: 2018-05-11 (Erik)

Author: E.C. van Berchum(TU Delft)

"""

import os
import sys
#sys.path.append(os.path.abspath('..'))
from src.FLORES_tools.Library.flood_simulation_model import (FloodSimModel, SimulationInput)
from datetime import date

path_Tools = os.path.abspath('..')
path_src = os.path.abspath(os.path.join(path_Tools,".."))
path_print = print(path_src)

from src.FLORES_tools.Library.simulation_data_beira import get_hydraulic_boundary_conditions, \
    get_region_layout, get_strategy
from timeit import default_timer as timer
from src.FLORES_tools.Library.simulation_definitions_beira import Impact
from src.FLORES_tools.Library.simulation_calculations_beira import run_hydraulic_calculation
import csv

# Preparation, load all external data
time_prep = timer()

dir_name_data = os.path.join(path_src, 'Projects/FLORES_beira/input_data').replace('\\','/')
#dir_name_data = path_current / 'Projects' / 'FLORES_beira' / 'input_data'
print(dir_name_data)
flores_sim = FloodSimModel()

flores_dem = 'TanDEM'
pop_source = 'ADFR_pop'
str_source = 'ADFR_str'
folder_exposure = 'exposure'
dam_curve_source = "damage_curves_JRC"
measures_source = 'flood_risk_reduction_measures'
rain_source = 'study_chiveve'
ss_source = 'SS'
climate_source = 'IPCC'
urban_source = 'GOV'

suffix_data = '.csv'
suffix_dam_curve = ".xlsx"
suffix_measures_data = '.csv'
suffix_scenarios_data = '.csv'
folder_hazard = 'hazards'
folder_scenarios = 'scenarios'

dem_folderpath = os.path.join(dir_name_data, 'schematization/', flores_dem).replace('\\','/')
dem_datafolder = os.path.join(dem_folderpath, 'data').replace('\\','/')
exposure_folderpath = os.path.join(dem_datafolder, folder_exposure).replace('\\','/')
pop_folder = os.path.join(exposure_folderpath,pop_source).replace('\\','/')
str_folder = os.path.join(exposure_folderpath, str_source).replace('\\','/')
hazard_folderpath = os.path.join(dir_name_data, folder_hazard).replace('\\','/')
scenarios_folderpath = os.path.join(dir_name_data, folder_scenarios).replace('\\','/')


basin_datafile = str(os.path.join(dem_datafolder, flores_dem + '_basins' + suffix_data)).replace('\\','/')
LOD_datafile = str(os.path.join(dem_datafolder, flores_dem + '_lines_of_defense' + suffix_data)).replace('\\','/')
basin_borders_datafile = str(os.path.join(dem_datafolder, flores_dem + '_basin_borders' + suffix_data)).replace('\\','/')
basin_drainage_datafile = str(os.path.join(dem_datafolder, flores_dem + '_basin_drainage' + suffix_data)).replace('\\','/')
pop_datafile = str(os.path.join(pop_folder, pop_source + suffix_data)).replace('\\','/')
str_datafile = str(os.path.join(str_folder, str_source + suffix_data)).replace('\\','/')
dam_curve_file = str(os.path.join(dir_name_data, 'schematization/' + dam_curve_source + suffix_dam_curve)).replace('\\','/')
measures_datafile = str(os.path.join(dir_name_data,'measures/' + measures_source + suffix_measures_data)).replace('\\','/')
rain_datafile = str(os.path.join(hazard_folderpath, 'rainfall_' + rain_source + suffix_data)).replace('\\','/')
ss_datafile = str(os.path.join(hazard_folderpath, 'surge_' + ss_source + suffix_data)).replace('\\','/')
climate_scenarios_data = str(os.path.join(scenarios_folderpath,'climate_' + climate_source + suffix_scenarios_data)).replace('\\','/')
urban_scenarios_data = str(os.path.join(scenarios_folderpath, 'urban_dev_' + urban_source + suffix_scenarios_data)).replace('\\','/')


flores_sim.save_source('basins', basin_datafile)
flores_sim.save_source('layers', LOD_datafile)
flores_sim.save_source('basin_borders', basin_borders_datafile)
flores_sim.save_source('basin_drainage', basin_drainage_datafile)
flores_sim.save_source('population', pop_datafile)
flores_sim.save_source('structures', str_datafile)
flores_sim.save_source('damage_curves', dam_curve_file)
flores_sim.save_source('measures', measures_datafile)
flores_sim.define_active_measures(CD_1=True,
                                  CD_2=True,
                                  CD_3=True,
                                  CD_4=True,
                                  SM_1=True,
                                  DR_1=True,
                                  DR_2=True,
                                  RT_1=True,
                                  RT_2=True,
                                  EM_1=True,
                                  EM_2=True
                                  )
flores_sim.save_source('hazard_rain', rain_datafile)
flores_sim.save_source('hazard_surge', ss_datafile)
flores_sim.save_source('climate_scenarios', climate_scenarios_data)
flores_sim.save_source('urban_development_scenarios', urban_scenarios_data, 'yes')

time_prep_end = timer()

print('preparation time is {} seconds'.format(time_prep_end - time_prep))

time_start = timer()
sim_input = SimulationInput(return_period_storm_surge=10,
                            return_period_rainfall=0,
                            flood_risk_reduction_strategy=[],
                            future_scenario="low",
                            structural_heights={}
                            )

return_period_storm = sim_input.ReturnPeriodStormSurge
return_period_rain = sim_input.ReturnPeriodRainfall
input_scenario_climate = sim_input.Scenario
input_scenario_development = 'none'
chosen_measures = sim_input.ChosenStrategy

# start of model, loads simulation-specific data
hydraulic = get_hydraulic_boundary_conditions(flores_sim.HydraulicConditionsMaster, return_period_storm,
                                                   return_period_rain,
                                                   input_scenario_climate)
region_layout = get_region_layout(flores_sim.BasinsMaster, flores_sim.LayersMaster, input_scenario_development)
strategy = get_strategy(master=flores_sim.AllMeasures, region_layout=region_layout,
                        chosen_measures=chosen_measures, structural_measures_height=sim_input.StructuralHeights)
impact = Impact()

# Builds and correctly names the scenarios
for sequence in [1, 3]:
    region_layout.Layers[sequence].get_scenarios()
strategy.get_list_scenarios(region_layout)
time_load = timer()

# Hydraulic calculations, runs entire hydraulic simulation (pluvial and storm surge flooding)
[inflow_rain, inflow_storm, outflow_drain, outflow_infiltration, volume_retention] = run_hydraulic_calculation(region_layout, hydraulic, strategy)
time_hydraulic = timer()
strategy.get_probabilities(region_layout, hydraulic)

# Calculate cost of construction and repair
total_cost = strategy.get_construction_costs()

# Impact calculations, calculates expected damages and exposed population per basin and in total
impact.run_impact_calculations(region_layout, strategy, flores_sim.DamageCurves)
time_impact = timer()

#risk_reduction = impact.TotalExpectedDamage
#construction_cost = total_cost
#affected_pop_reduction = impact.TotalExpectedExposedPop

print('simulation succesful')
print('loading stage time is {} seconds'.format(time_load - time_start))
print('hydraulic computation time is {} seconds'.format(time_hydraulic - time_load))
print('impact calculation computation time is {} seconds'.format(time_impact - time_hydraulic))
print('total damage is: ', impact.TotalExpectedDamage)
print('total construction cost is: ', total_cost)
print('total affected population is: ', sum(impact.ExpectedBasinExposedPop.values()))
print('precentage people affected is : ',  sum(impact.ExpectedBasinExposedPop.values()) / 530000)

total_volume_in_system = {}
total_in = {}
total_end = {}
for scenario in strategy.ListScenarios:
    total_volume_in_system[scenario] = 0
    total_in[scenario] = 0
    total_end[scenario] = 0
    for basin in region_layout.Basins:
        total_volume_in_system[scenario] += region_layout.Basins[basin].HeightToVolume(float(
            region_layout.Basins[basin].ScenarioWaterLevels[scenario][-1] + region_layout.Basins[basin].Contours[
                0].MinHeight))
    total_in[scenario] = inflow_storm[scenario] + inflow_rain[scenario]
    total_end[scenario] = total_volume_in_system[scenario] + outflow_drain[scenario] + outflow_infiltration[scenario] \
                          + volume_retention[scenario]
#     print("scenario: {}: total inflow rain: {}, inflow storm {}, outflow drain: {}, outflow infiltration: {} "
#           "left in system: {}.  total retention: {}. Total in: {}, Total end: {}".
#           format(scenario, inflow_rain[scenario], inflow_storm[scenario], outflow_drain[scenario],
#                  outflow_infiltration[scenario], total_volume_in_system[scenario], volume_retention[scenario],
#                  total_in[scenario], total_end[scenario]))
#print('volume in system from calc: {}, volume in system after: {}'.format(volume_in_system, total_volume_in_system))

region_layout.show_results(1, 2, 5, 8, 9, 10, 11, 12, 13, strategy,
                           return_period_storm, return_period_rain)

impact.show_results(region_layout, inflow_rain, inflow_storm, outflow_drain, outflow_infiltration,
                    volume_retention, total_volume_in_system, total_in, total_end, return_period_storm, return_period_rain)

today = date.today()
i = 1
results_path = os.path.join(path_src,
                            'Projects/FLORES_beira/data/Simulations/Simulation_Beira_storm_{}_rain_{}_{}_run_1.csv'.format(return_period_storm, return_period_rain, today))
if os.path.exists(results_path):
    i +=1
    results_path = os.path.join(path_src,
                            'Projects/FLORES_beira/data/Simulations/Simulation_Beira_storm_{}_rain_{}_{}_run_{}.csv'.format(
                                return_period_storm, return_period_rain, today, i))

with open(results_path, 'w', newline='') as csv_file:
    spamwriter = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(['Basin_ID', 'Inundation level (Scenario 1.1)', 'Expected damage', 'Expected affected population'])
for basin in impact.ExpectedBasinDamages:
    spamwriter.writerow([basin, region_layout.Basins[basin].ScenarioMaxWaterLevels['1.1']
                         + region_layout.Basins[basin].Contours[0].MinHeight, impact.ExpectedBasinDamages[basin],
                         impact.ExpectedBasinExposedPop[basin]])

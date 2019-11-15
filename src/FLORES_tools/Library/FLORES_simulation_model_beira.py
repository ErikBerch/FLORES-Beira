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

# from __future__ import (absolute_import, division,
#                        print_function, unicode_literals)
from simulation_data_beira_V1_1 import load_hydraulic_conditions, get_hydraulic_boundary_conditions, \
    load_basins, load_measures, get_region_layout, get_strategy, load_layers, \
    load_damage_curves, load_basin_borders, load_drainage_capacities
from timeit import default_timer as timer
from simulation_definitions_beira_V1_1 import Impact
from simulation_calculations_beira_V1_1 import run_hydraulic_calculation


import csv

# Preparation, load all external data
time_prep = timer()
basins_master = load_basins("input_data/region_layout_basins_V1_1.csv",
                            "input_data/urban_development_scenarios_V1_1.csv")  # Regional Layout
layers_master = load_layers("input_data/region_layout_layers_V1_1.csv", basins_master)
all_measures_master = load_measures("input_data/flood_risk_reduction_measures_V1_2.csv")
hydraulic_conditions_master = load_hydraulic_conditions("input_data/hydraulic_boundary_conditions_surge_beira_V1_1.csv",
                                                        "input_data/hydraulic_boundary_conditions_rain_beira_V1_1.csv")

damage_curves = load_damage_curves('input_data/copy_of_global_flood_depth-damage_functions__30102017.xlsx', 'AFRICA',
                                   'Mozambique', 'Object based', 1.30)
load_basin_borders(basins_master, "input_data/region_layout_basin_borders_V1_1.csv")
drainage_master = load_drainage_capacities(basins_master, 'input_data/basins_drainage_V1_1.csv', small_channel=6,
                                           mid_channel=10, large_channel=35, low_drain=0, mid_drain=2, high_drain=4)

# Simulation based variables on strategy and the situation
# return_period_storm = 0
# return_period_rain = 100
# struc_measure_coast_1 = 'none'  # 'Heighten dunes east','Sand supplements east', None
# struc_measure_coast_2 = 'none'  # 'Heighten dunes west','Floodwall west'', None
# struc_measure_inland_1 = 'none'  # 'Heighten inland road', None
# struc_measure_inland_2 = 'none'                  # None
# drainage_measure_1 = 'none'  # 'Second phase drainage', None
# drainage_measure_2 = 'none'  # 'Microdrainage, None
# retention_measure_1 = 'none'  # 'East retention', None
# retention_measure_2 = 'none'   # 'Chota retention', None
# emergency_measure_1 = 'none'  # 'Improve evacuation', None
# emergency_measure_2 = 'none'   # 'Early warning system', None
# h_struc_measure_coast_1 = 8.2731906  # 8-12
# h_struc_measure_coast_2 = 11.872320  # 8-12
# h_struc_measure_inland_1 = 11.3344123  # 7-10
# h_struc_measure_inland_2 = 0     # 0
# input_scenario_climate = 'none'  # 'high','low', 'none'
# input_scenario_development = 'none'  # 'high','low', 'none'

# Simulation based variables on strategy and the situation
return_period_storm = 50
return_period_rain = 10
struc_measure_coast_1 = 'none'  # 'Heighten dunes east','Sand supplements east', None
struc_measure_coast_2 = 'none'  # 'Heighten dunes west','Floodwall west'', None
struc_measure_inland_1 = 'none'  # 'Heighten inland road', None
struc_measure_inland_2 = 'none'                   # None
drainage_measure_1 = 'none'  # 'Second phase drainage', None
drainage_measure_2 = 'none'  # 'Microdrainage, None
retention_measure_1 = 'none' # 'East retention', None
retention_measure_2 = 'none'   # 'Chota retention', None
emergency_measure_1 = 'Improve_evacuation'  # 'Improve evacuation', None
emergency_measure_2 = 'none'   # 'Early warning system', None
local_measure_1 = 'Strengtening_houses'   # ' Strengtening_houses','none'
h_struc_measure_coast_1 = 10.5  # 8-12
h_struc_measure_coast_2 = 9  # 8-12
h_struc_measure_inland_1 = 0  # 7-10
h_struc_measure_inland_2 = 0     # 0
input_scenario_climate = 'high'  # 'high','low', 'none'
input_scenario_development = 'none'  # 'high','low', 'none'


# start of model, loads simulation-specific data
time_start = timer()
hydraulic = get_hydraulic_boundary_conditions(hydraulic_conditions_master, return_period_storm, return_period_rain,
                                              input_scenario_climate)
region_layout = get_region_layout(basins_master, layers_master, input_scenario_development)
strategy = get_strategy(all_measures_master, region_layout,
                        [struc_measure_coast_1, struc_measure_coast_2, struc_measure_inland_1, struc_measure_inland_2,
                         drainage_measure_1, drainage_measure_2, retention_measure_1, retention_measure_2, local_measure_1,
                         emergency_measure_1, emergency_measure_2],
                        [h_struc_measure_coast_1, h_struc_measure_coast_2, h_struc_measure_inland_1,
                         h_struc_measure_inland_2])

# Builds and correctly names the scenarios
for sequence in [1, 3]:
    region_layout.Layers[sequence].get_scenarios()
strategy.get_list_scenarios(region_layout)
time_load = timer()
impact = Impact()
# Hydraulic calculations, runs entire hydraulic simulation (pluvial and storm surge flooding)
[inflow_rain, inflow_storm, outflow_drain, outflow_infiltration, volume_retention] = run_hydraulic_calculation(
    region_layout, hydraulic, strategy)
time_hydraulic = timer()

strategy.get_probabilities(region_layout, hydraulic)
total_cost = strategy.get_construction_costs()

# Impact calculations, calculates expected damages and exposed population per basin and in total
impact.run_impact_calculations(region_layout, strategy, damage_curves)
time_impact = timer()

# After model
print('preparation time is {} seconds'.format(time_start - time_prep))
print('loading stage time is {} seconds'.format(time_load - time_start))
print('hydraulic computation time is {} seconds'.format(time_hydraulic - time_load))
print('impact calculation computation time is {} seconds'.format(time_impact - time_hydraulic))

total_volume_in_system = {}
total_in = {}
total_end = {}
for scenario in strategy.ListScenarios:
    total_volume_in_system[scenario] = 0
    total_in[scenario] = 0
    total_end[scenario] = 0
    for basin in region_layout.Basins:
        total_volume_in_system[scenario] += region_layout.Basins[basin].HeightToVolume(float(
            region_layout.Basins[basin].ScenarioWaterLevels[scenario][-2] + region_layout.Basins[basin].Contours[
                0].MinHeight))

    total_in[scenario] = inflow_storm[scenario] + inflow_rain[scenario]
    total_end[scenario] = total_volume_in_system[scenario] + outflow_drain[scenario] + outflow_infiltration[scenario] \
                          + volume_retention[scenario]
    print("scenario: {}: total inflow rain: {}, inflow storm {}, outflow drain: {}, outflow infiltration: {} "
          "left in system: {}.  total retention: {}. Total in: {}, Total end: {}".
          format(scenario, inflow_rain[scenario], inflow_storm[scenario], outflow_drain[scenario],
                 outflow_infiltration[scenario], total_volume_in_system[scenario], volume_retention[scenario],
                 total_in[scenario], total_end[scenario]))

#print('total damage is: ', impact.TotalExpectedDamage)
#print('total construction cost is: ', total_cost)
#print('total affected population is: ', sum(impact.ExpectedBasinExposedPop.values()))

#print('precentage people affected is : ',  sum(impact.ExpectedBasinExposedPop.values()) / 530000)

region_layout.show_results(2, 14, 20, 28, 29, 37, 39, 41, 43, strategy,
                           return_period_storm, return_period_rain, struc_measure_coast_1, struc_measure_coast_2, struc_measure_inland_1,
                           drainage_measure_1, drainage_measure_2, retention_measure_1, retention_measure_2, emergency_measure_1, emergency_measure_2,
                           h_struc_measure_coast_1, h_struc_measure_coast_2, h_struc_measure_inland_1)
#
impact.show_results(region_layout, inflow_rain, inflow_storm, outflow_drain, outflow_infiltration,
                     total_volume_in_system, total_in, total_end, return_period_storm, return_period_rain,
                     struc_measure_coast_1, struc_measure_coast_2, struc_measure_inland_1, drainage_measure_1,
                     drainage_measure_2, retention_measure_1, retention_measure_2,
                      emergency_measure_1, emergency_measure_2,
                     h_struc_measure_coast_1, h_struc_measure_coast_2, h_struc_measure_inland_1)



# with open('Results/Simulation_Storm0_Rain5_1209_with_half_drainage_better_curves.csv', 'w', newline='') as csv_file:
#     spamwriter = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['Basin_ID', 'Inundation level (Scenario 1.1)',
#                          'Expected damage', 'Expected affected population'])
#     for basin in impact.ExpectedBasinDamages:
#         spamwriter.writerow([basin, region_layout.Basins[basin].ScenarioMaxWaterLevels['1.1']
#                              + region_layout.Basins[basin].Contours[0].MinHeight, impact.ExpectedBasinDamages[basin],
#                              impact.ExpectedBasinExposedPop[basin]])


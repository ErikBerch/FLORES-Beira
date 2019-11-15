from __future__ import (absolute_import, division,
                    print_function, unicode_literals)

from  Library.simulation_calculations_beira_V1_1 import run_hydraulic_calculation
import Library.simulation_definitions_beira_V1_1 as modeldef
import Library.simulation_data_beira_V1_1 as data
import pdb

basins_master = data.load_basins("./Library/input_data/region_layout_basins_V1_1.csv",
                                 "./Library/input_data/urban_development_scenarios_V1_1.csv")  # Regional Layout
layers_master = data.load_layers("./Library/input_data/region_layout_layers_V1_1.csv", basins_master)
all_measures_master = data.load_measures("./Library/input_data/flood_risk_reduction_measures_V1_1.csv")

#print(all_measures_master)

hydraulic_conditions_master = data.load_hydraulic_conditions("./Library/input_data/hydraulic_boundary_conditions_surge_beira_V1_1.csv",
                                                        "./Library/input_data/hydraulic_boundary_conditions_rain_beira_V1_1.csv")

damage_curves = data.load_damage_curves('./Library/input_data/copy_of_global_flood_depth-damage_functions__30102017.xlsx', 'AFRICA',
                                   'Mozambique', 'Object based', 1.30)
data.load_basin_borders(basins_master, "./Library/input_data/region_layout_basin_borders_V1_1.csv")
drainage_master = data.load_drainage_capacities(basins_master, './Library/input_data/basins_drainage_V1_1.csv', small_channel=6,
                                           mid_channel=10, large_channel=35, low_drain=0, mid_drain=2, high_drain=4)

def flood_simulation_model(return_period_storm = 0,    # 0, 2, 5, 10, 50, 100
                             return_period_rain = 0,    # 0, 2, 5, 10, 50, 100
                             struc_measure_coast_1 = 'none',  # 'Heighten dunes east','Sand supplements east', None
                             struc_measure_coast_2 = 'none',  # 'Heighten dunes west','Floodwall west'', None
                             struc_measure_inland_1 = 'none',  # 'Heighten inland road', None
                             #struc_measure_inland_2 = 'none',
                             drainage_measure_1 = 'none',  # 'Second phase drainage', None
                             drainage_measure_2 = 'none',   # 'Microdrainage, None
                             drainage_measure_3 = 'none',
                             drainage_measure_4 = 'none',
                             retention_measure_1 = 'none',  # 'East retention', None
                             retention_measure_2 = 'none',    # 'Maraza retention, None
                             emergency_measure_1 = 'none',  # 'Improve evacuation', None
                             emergency_measure_2 = 'none',    # 'Early warning system', None
                             local_measure_1 = 'none',        #  'Strengthening_houses', None
                             local_measure_2 = 'none',       # 'Prevent_settlement_vulnerable_areas', None
                             h_struc_measure_coast_1 = 0,  # 8-12
                             h_struc_measure_coast_2 = 0,  # 8-12
                             h_struc_measure_inland_1 = 0,  # 7-10
                             h_struc_measure_inland_2 = 0,
                             input_scenario_climate = 'none',  # 'high','low', 'none'
                             input_scenario_development = 'none'  # 'high','low', 'none'
                            ):

    # added so we can add scenario information into the replicator
    if return_period_rain == 'INFO':
        scenario_info = str(input_scenario_climate) + ',' + str(input_scenario_development)
        return [scenario_info]*3
    
    # start of model, loads simulation-specific data
    hydraulic = data.get_hydraulic_boundary_conditions(hydraulic_conditions_master, return_period_storm, return_period_rain,
                                                  input_scenario_climate)
    region_layout = data.get_region_layout(basins_master, layers_master, input_scenario_development)
# =============================================================================
#     strategy = data.get_strategy(all_measures_master, region_layout,
#                             [struc_measure_coast_1, struc_measure_coast_2, struc_measure_inland_1, struc_measure_inland_2,
#                              drainage_measure_1, drainage_measure_2, retention_measure_1, retention_measure_2,
#                              emergency_measure_1, emergency_measure_2],
#                              #local_measure_1,local_measure_2], # LOCAL MEASURES TOEGEVOEGD, goed ????
#                             [h_struc_measure_coast_1, h_struc_measure_coast_2, h_struc_measure_inland_1,
#                              h_struc_measure_inland_2])
# =============================================================================
    #pdb.set_trace()
    chosen_measures=[struc_measure_coast_1, struc_measure_coast_2, struc_measure_inland_1, #struc_measure_inland_2,
                                  drainage_measure_1, drainage_measure_2, drainage_measure_3, drainage_measure_4, retention_measure_1, retention_measure_2,
                                  emergency_measure_1, emergency_measure_2,
                                  local_measure_1,local_measure_2]
    #print(chosen_measures)
    structural_measures_height=[h_struc_measure_coast_1, h_struc_measure_coast_2, h_struc_measure_inland_1,
                                  h_struc_measure_inland_2]
    strategy = data.get_strategy(master=all_measures_master, region_layout=region_layout, 
                                 chosen_measures=chosen_measures, structural_measures_height=structural_measures_height)
    impact = modeldef.Impact()

    # Builds and correctly names the scenarios
    for sequence in [1, 3]:
        region_layout.Layers[sequence].get_scenarios()
    strategy.get_list_scenarios(region_layout)

    # Hydraulic calculations, runs entire hydraulic simulation (pluvial and storm surge flooding)
    run_hydraulic_calculation(region_layout, hydraulic, strategy)

    strategy.get_probabilities(region_layout, hydraulic)

    # Calculate cost of construction and repair
    total_cost = strategy.get_construction_costs()

    # Impact calculations, calculates expected damages and exposed population per basin and in total
    impact.run_impact_calculations(region_layout, strategy, damage_curves)
    risk_reduction = impact.TotalExpectedDamage
    construction_cost = total_cost
    #pdb.set_trace()
    affected_pop_reduction = impact.TotalExpectedExposedPop
    
    return risk_reduction, construction_cost, affected_pop_reduction
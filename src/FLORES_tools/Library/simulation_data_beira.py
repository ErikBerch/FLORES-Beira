# -*- coding: utf-8 -*-
"""
This file is part of the simulation of the MODOS-model. The MODOS-model can be used to compare various flood risk
reduction strategies and consists of a Simulation part and an Evaluation part. the Simulation requires four .py files:

- MODOS_simulation_model_<Project>_<Version>.py
    - simulation_definitions_<Project>_<Version>.py
    - simulation_data_<Project>_<Version>.py
    - simulation_calculations_<Project>_<Version>.py

This file lists project-specific data, which is used to fill the objects as defined in 'simulation_defnitions'. The data
includes information on the region lay-out, the flood risk reduction measures and the hydraulic boundary conditions.

Project: Beira, Mozambique
Last updated: 2018-05-11(Erik)

Author: E.C. van Berchum(TU Delft)

"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .simulation_definitions_beira import (Basin, HydraulicBoundaryConditions, LineOfDefense, LOD_Land, LOD_Water,
                                              NatureBased, EmergencyMeasure, DrainageMeasure, RetentionMeasure,
                                              RegionLayout, ProtectedArea, UnprotectedArea, StructuralMeasureLand,
                                              StructuralMeasureWater, FRRStrategy, Contour, DamageFunctions)
from .simulation_calculations_beira import tolist
import ast
import copy
import pandas as pd
from scipy import interpolate
import numpy as np


def load_hydraulic_conditions(file_surge, file_rain):

    hydraulic_conditions_master = {}
    surge_master = pd.read_csv(file_surge, sep=';', header=0)
    rain_master = pd.read_csv(file_rain, sep=';', header=0)
    surge_master_dropped = surge_master.dropna()
    rain_master_dropped = rain_master.dropna()
    timestep_model = 0.1  # hours

    for i in range(len(surge_master_dropped)):
        for j in range(len(rain_master_dropped)):
            tmp_surge = surge_master_dropped.loc[i]
            tmp_rain = rain_master_dropped.loc[j]
            key = str(int(tmp_surge['return period storm'])) + ';' + str(int(tmp_rain['return period rain']))             # name of the storm/rain combination, e.g. '100;10'
            length_series = int(tmp_surge['storm duration']/timestep_model + 1)
            surge_climate_scenarios = {'none': 0}
            rain_climate_scenarios = {'none': 1}
            rain_series = [int(tmp_rain['Rain intensity_1'])] * length_series
            wind_series = ['N'] * length_series
            for input_climate_scenario in ['low', 'high']:
                surge_climate_scenarios[input_climate_scenario] = tmp_surge['Climate_scenario_' + input_climate_scenario]
                rain_climate_scenarios[input_climate_scenario] = tmp_rain['Climate_scenario_' + input_climate_scenario]

            hydraulic_conditions_master[key] = HydraulicBoundaryConditions(
                tmp_surge['return period storm'], tmp_rain['return period rain'],
                rain_series, wind_series, tmp_surge['maximum surge'], tmp_rain['Rain intensity_2'],
                tmp_surge['wave height'], tmp_surge['wind velocity'], tmp_surge['storm duration'], timestep_model,
                tmp_surge['normal tidal amplitude'], tmp_surge['MSL'], surge_climate_scenarios, rain_climate_scenarios)

    return hydraulic_conditions_master


def get_hydraulic_boundary_conditions(hydraulic_conditions_master, return_period_storm, return_period_rain, climate_scenario):
    """
    This function chooses the hydraulic boundary conditions from the input data, based on return periods of storm and
    rain.

    :param hydraulic_conditions_master: input data, comes from load_hydraulic_conditions
    :param return_period_storm:
    :param return_period_rain:
    :return:
    """
    if not climate_scenario:
        print('wrong input, use "none" instead of None for climate scenario')
    key = str(int(return_period_storm)) + ';' + str(int(return_period_rain))
    hydraulic_boundary_conditions = create_copy(hydraulic_conditions_master[key])
    hydraulic_boundary_conditions.apply_climate_scenario(climate_scenario)

    return hydraulic_boundary_conditions


def get_region_layout(basins_master, layers_master, develop_scenario):

    if not develop_scenario:
        print('wrong input, use "none" instead of None for development scenario')
    basins_dict = create_copy(basins_master[develop_scenario])
    layers_dict = create_copy(layers_master)
    region_layout = RegionLayout(basins_dict, layers_dict)

    return region_layout


def load_basins(file, file_development):

    basins_source = pd.read_csv(file, sep=';', header=0)
    development_scenario_source = pd.read_csv(file_development, sep=';', header=0)
    development_scenario_master = {}
    for row in range(0, len(development_scenario_source)):
        tmp_development = development_scenario_source.loc[row]
        development_scenario_master[tmp_development['Name']] = {'DEV_1':tmp_development['Factor_DEV_1'], 'DEV_2':tmp_development['Factor_DEV_2'], 'DEV_3':tmp_development['Factor_DEV_3'], 'DEV_5':tmp_development['Factor_DEV_5'], 'DEV_7':tmp_development['Factor_DEV_7']}
    basins_master = {}

    for development_scenario in development_scenario_master:
        basins_master[development_scenario] = {}
        for row in range(0, len(basins_source)):
            tmp_basin = basins_source.loc[row]
            #landuse_dictionary = {}
            landuse_value_dictionary = {}
            for landusetype in ['DEV_1', 'DEV_2', 'DEV_3', 'DEV_5', 'DEV_7']:
                #landuse_dictionary[landusetype] = tmp_basin[landusetype + '_Area']                  # {type,value of land use}
                landuse_value_dictionary[landusetype] = tmp_basin[landusetype + '_Value'] * development_scenario_master[development_scenario][landusetype]
            if int(tmp_basin['Basin_ID']) == len(basins_master[development_scenario]) - 1 and len(basins_master[development_scenario]) != 0:
                basins_master[development_scenario][int(tmp_basin['Basin_ID'])].SurfaceArea += tmp_basin['area (m2)']
                basins_master[development_scenario][int(tmp_basin['Basin_ID'])].Contours.append(Contour(int(tmp_basin['Contour_ID']),tmp_basin['Height_min'],
                                                                                    tmp_basin['area (m2)'], landuse_value_dictionary, tmp_basin['Population'] *
                                                                                                        development_scenario_master[development_scenario][landusetype]))
            else:
                basins_master[development_scenario][int(tmp_basin['Basin_ID'])] = Basin(str(int(tmp_basin['Basin_ID'])), surfacearea=tmp_basin['area (m2)'],
                                                                  contours=[Contour(int(tmp_basin['Contour_ID']),tmp_basin['Height_min'],
                                                                                    tmp_basin['area (m2)'], landuse_value_dictionary, tmp_basin['Population']*
                                                                           development_scenario_master[development_scenario][landusetype])])
        for basin in basins_master[development_scenario]:
            basins_master[development_scenario][basin].get_infiltration_rate()
            basins_master[development_scenario][basin].get_volume_inundation_curve()
    return basins_master


def load_layers(file, basins_master):

    layers_source = pd.read_csv(file, sep=';', header=0)
    layers_master = {}
    for row in range(0, len(layers_source)):
        tmp_layer = layers_source.loc[row]
        if tmp_layer['Type'] == 'Line of Defense':
            fd_locations = {}
            for defense in range(1, int(tmp_layer['Number of flood defenses'])+1):
                code = 'FD' + str(defense) + ' '
                if tmp_layer[code + 'Type'] == 'Land':
                    basin_codes = tolist(tmp_layer[code + 'Basin codes'], ';')
                    basin_widths = tolist(tmp_layer[code + 'Basin widths'], ';')
                    basin_codes_widths_coupled = dict(zip(basin_codes, basin_widths))
                    incomingbasin = int(tmp_layer[code + 'Incoming basin'])
                    fd_locations[tmp_layer[code + 'Name']] = LOD_Land(tmp_layer[code + 'Name'], tmp_layer[code + 'Type'], tmp_layer['Sequence'],
                                                                         basin_codes_widths_coupled, tmp_layer[code + 'Width'], tmp_layer[code + 'Height'], incoming_basin=incomingbasin)
                elif tmp_layer[code + '_type'] == 'Water':
                    basin_codes = tolist(tmp_layer[code + 'Basin codes'], ';')
                    fd_locations[tmp_layer[code + 'Name']] = LOD_Water(tmp_layer[code + 'Name'], tmp_layer[code + 'Type'], basin_codes, tmp_layer[code + 'Width'], tmp_layer[code + 'Depth'], tmp_layer[code + 'Length'], incoming_basin=tmp_layer[code + 'Incoming basin'])
                else:
                    print('wrong flood defense type chosen:' + str(tmp_layer['Name']))

            layers_master[tmp_layer['Sequence']] = LineOfDefense(tmp_layer['Name'], tmp_layer['Type'],
                                                                 tmp_layer['Sequence'], fdlocations=fd_locations)

        elif tmp_layer['Type'] == 'Protected area':
            basin_codes = tolist(tmp_layer['Basin codes'], ';')
            layers_master[tmp_layer['Sequence']] = ProtectedArea(tmp_layer['Name'], tmp_layer['Type'],
                                                                 tmp_layer['Sequence'], basin_codes)
        elif tmp_layer['Type'] == 'Unprotected area':
            basin_codes = tolist(tmp_layer['Basin codes'], ';')
            layers_master[tmp_layer['Sequence']] = UnprotectedArea(tmp_layer['Name'], tmp_layer['Type'],
                                                                   tmp_layer['Sequence'], basin_codes)
        else:
            print("Wrong layer type chosen: " + str(tmp_layer['Name']))

    # load basin widths into basin objects
    for sequence in layers_master:
        if layers_master[sequence].Type == 'Line of Defense':
            for location in layers_master[sequence].FDLocations:
                for basin in layers_master[sequence].FDLocations[location].BasinCodesWidths:
                    for development_scenario in basins_master:
                        basins_master[development_scenario][basin].Width = layers_master[sequence].FDLocations[location].BasinCodesWidths[basin]

    return layers_master


def load_measures(file):
    """"
    get the measures from the excel file and load them into the python script. This should be done before the
    simulation. In the start of the simulation, a copy should be loaded. In this case, functions that change
    characteristics of the measures do not affect other simulations.

    example of use:
    all_measures_master = load_measures('case_study')
    """
    measures_source = pd.read_csv(file, sep=';', header=0)
    measures_master = {}
    for row in range(0, len(measures_source)):
        tmp_measure = measures_source.loc[row]
        if tmp_measure['Type'] == 'Structural':
            tmp_heights_min_max = tolist(tmp_measure['Potential_heights_min_max'], ';', float)
            tmp_potential_heights = np.arange( tmp_heights_min_max[0], tmp_heights_min_max[1] + 0.5, 0.5).tolist()
            if tmp_measure['STR_Land/Water'] == 'Land':
                measures_master[tmp_measure['Code']] = StructuralMeasureLand(tmp_measure['Name'], tmp_measure['Layer'],
                                                                             tmp_measure['Location'], float(tmp_measure['Constant costs']), float(tmp_measure['Variable costs']),
                                                                             tmp_potential_heights,irribaren=float(tmp_measure['STR_irribaren']))
            elif tmp_measure['STR_Land/Water'] == 'Water':
                measures_master[tmp_measure['Code']] = StructuralMeasureWater(tmp_measure['Name'], tmp_measure['Layer'],
                                                                              tmp_measure['Location'], float(tmp_measure['Constant costs']), float(tmp_measure['Variable costs']),
                                                                              tmp_potential_heights, irribaren=float(tmp_measure['STR_irribaren']))
            else:
                print("wrong structural location chosen: " + str(tmp_measure['Measure']))
        elif tmp_measure['Type'] == 'Nature-based solution':
            measures_master[tmp_measure['Code']] = NatureBased(tmp_measure['Name'], tmp_measure['Location'],
                                                                  tmp_measure['NBS_effect'], tmp_measure['NBS_factor'],
                                                                  tmp_measure['Constant costs'])
        elif tmp_measure['Type'] == 'Emergency':
            change_emergency_codes = tolist(tmp_measure['Location'], ';')
            change_emergency_factors = [tmp_measure['EM_factor']] * len(change_emergency_codes)
            change_emergency_dict = dict(zip(change_emergency_codes, change_emergency_factors))
            measures_master[tmp_measure['Code']] = EmergencyMeasure(tmp_measure['Name'], change_emergency_dict,
                                                                         tmp_measure['EM_effect'],
                                                                         float(tmp_measure['Constant costs']))
        elif tmp_measure['Type'] == 'Drainage':
            change_drainage_codes = tolist(tmp_measure['Location'], ';')
            change_drainage_capacity = tolist(tmp_measure['New_drainage_capacity'], ';')
            if len(change_drainage_capacity) == 1:
                change_drainage_capacity = change_drainage_capacity * len(change_drainage_codes)
            change_drainage_dict = dict(zip(change_drainage_codes, change_drainage_capacity))
            measures_master[tmp_measure['Code']] = DrainageMeasure(tmp_measure['Name'], change_drainage_dict,
                                                                   tmp_measure['Constant costs'])

        elif tmp_measure['Type'] == 'Retention':
            change_retention_codes = tolist(tmp_measure['Location'], ';')
            change_retention_capacity = tolist(tmp_measure['Retention_capacity'], ';')
            if len(change_retention_capacity) == 1:
                change_retention_capacity = change_retention_capacity * len(change_retention_codes)
            change_retention_dict = dict(zip(change_retention_codes, change_retention_capacity))
            measures_master[tmp_measure['Code']] = RetentionMeasure(tmp_measure['Name'], change_retention_dict,
                                                                    tmp_measure['Constant costs'])

        else:
            print("wrong type input for measure: " + str(tmp_measure['Name']))

    return measures_master


def create_copy(master):
    """
    This function creates a copy of the master database to use in the simulation. This protects the database (region
    layout / hydraulic boundary conditions / measures) from changes made during a simulation.

    :param master: database to copy
    :return:
    """
    copy_database = copy.deepcopy(master)
    return copy_database


def get_strategy(master, region_layout, chosen_measures, structural_measures_height):

    all_measures_copy = copy.deepcopy(master)
    strategy = FRRStrategy(chosen_measures, all_measures_copy)
    strategy.get_measures(structural_measures_height, region_layout.Layers)

    # load used measure into basins and layers
    for sequence in region_layout.Layers:
        if region_layout.Layers[sequence].Type == 'Line of Defense':
            for location in region_layout.Layers[sequence].FDLocations:
                for measure in strategy.StructuralMeasures:
                    if strategy.StructuralMeasures[measure].Location == location and strategy.StructuralMeasures[measure].Layer == region_layout.Layers[sequence].Name:
                        region_layout.Layers[sequence].FDLocations[location].UsedMeasure = measure
                        for basin in region_layout.Layers[sequence].FDLocations[location].BasinCodesWidths:
                            region_layout.Basins[basin].UsedMeasure = strategy.StructuralMeasures[region_layout.Layers[sequence].FDLocations[location].UsedMeasure]
                        break

    for measure in strategy.DrainageMeasures:
        for basin in strategy.DrainageMeasures[measure].ChangeBasins:
            region_layout.Basins[basin].add_drainage_discharge(strategy.DrainageMeasures[measure].ChangeBasins[basin])

    for measure in strategy.RetentionMeasures:
        for basin in strategy.RetentionMeasures[measure].ChangeBasins:
            region_layout.Basins[basin].RetentionCapacity += strategy.RetentionMeasures[measure].ChangeBasins[basin]

    for measure in strategy.EmergencyMeasures:
        for basin in strategy.EmergencyMeasures[measure].ChangeBasins:
            for contour in region_layout.Basins[basin].Contours:
                contour.Population *= strategy.EmergencyMeasures[measure].ChangeBasins[basin]

    return strategy


def load_damage_curves(file, continent, country, modeltype, exchange):

    xls = pd.ExcelFile(file)
    columns_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # First 9 columns, damage functions for all continents
    df_damage_function = pd.read_excel(xls, 'Damage functions', header=2, usecols=columns_to_use)
    factors = {}
    maxvalues = {}

    df_damage_function_continent = df_damage_function[continent]
    factors['Residential'] = df_damage_function_continent[0:9]
    factors['Commercial'] = df_damage_function_continent[9:18]
    factors['Industrial'] = df_damage_function_continent[18:27]
    factors['Transport'] = df_damage_function_continent[27:36]
    factors['Infrastructure'] = df_damage_function_continent[36:45]
    factors['Agriculture'] = df_damage_function_continent[45:54]

    maxvalues['Residential'] = pd.read_excel(xls, 'MaxDamage-Residential', header=0, index_col=0)[modeltype][country]
    maxvalues['Commercial'] = pd.read_excel(xls, 'MaxDamage-Commercial', header=0, index_col=0)[modeltype][country]
    maxvalues['Industrial'] = pd.read_excel(xls, 'MaxDamage-Industrial', header=0, index_col=0)[modeltype][country]
    maxvalues['Agriculture'] = pd.read_excel(xls, 'MaxDamage-Agriculture', header=0, index_col=0)['Value Added/Hectare\n(avg 2008-2012)'][country]
    maxvalues['Infrastructure'] = pd.read_excel(xls, 'MaxDamage-Infrastructure', header=1,usecols=[0, 1, 2],index_col=0)['GDP per capita (2010 US$)'][country]
    maxvalues['Transport'] = maxvalues['Infrastructure']

    damage_curves = DamageFunctions(factors, maxvalues, exchange)

    return damage_curves

def load_basin_borders(basins, file):

    borders_source = pd.read_csv(file, sep=';', header=0)
    for development_scenario in basins:
        for row in range(0,len(borders_source)):
            tmp_border = borders_source.loc[row]
            basins[development_scenario][tmp_border['Basin 1']].BorderHeights[tmp_border['Basin 2']] = tmp_border['Border_height_mean_lowest25% (m)']
            basins[development_scenario][tmp_border['Basin 2']].BorderHeights[tmp_border['Basin 1']] = tmp_border['Border_height_mean_lowest25% (m)']

        for basin in basins[development_scenario]:
            for surrounding_basin in basins[development_scenario][basin].BorderHeights:
                basins[development_scenario][basin].SurroundingAreas[surrounding_basin] = basins[development_scenario][surrounding_basin].SurfaceArea

    return

# def load_basin_borders(basins, file):
#
#     borders_source = pd.read_csv(file, sep=';', header=0)
#     for development_scenario in basins:
#         for row in range(0,len(borders_source)):
#             tmp_border = borders_source.loc[row]
#             if tmp_border['Basin 1'] != 0 and tmp_border['Basin 2'] != 0:
#                 basins[development_scenario][tmp_border['Basin 1']].BorderHeights[tmp_border['Basin 2']] = tmp_border['Border_height_mean_lowest25% (m)']
#                 basins[development_scenario][tmp_border['Basin 2']].BorderHeights[tmp_border['Basin 1']] = tmp_border['Border_height_mean_lowest25% (m)']
#
#         for basin in basins[development_scenario]:
#             for surrounding_basin in basins[development_scenario][basin].BorderHeights:
#                 basins[development_scenario][basin].SurroundingAreas[surrounding_basin] = basins[development_scenario][surrounding_basin].SurfaceArea
#
#     return


def load_drainage_capacities(basins, file,  small_channel, mid_channel, large_channel, low_drain, mid_drain, high_drain):

    drainage_source = pd.read_csv(file, sep=';', header=0)
    for development_scenario in basins:
        for row in range(len(drainage_source)):
            tmp_basin = drainage_source.loc[row]
            basins[development_scenario][tmp_basin['Basin_ID']].ExitChannel = False
            basins[development_scenario][tmp_basin['Basin_ID']].RetentionCapacity = float(tmp_basin['Retention'])
            basins[development_scenario][tmp_basin['Basin_ID']].DrainsToBasin = tmp_basin['Drains to basin']
            if tmp_basin['Drainage channel'] == 'yes':
                basins[development_scenario][tmp_basin['Basin_ID']].DrainageChannel = True
                channel_size = tmp_basin['channel size']
                if channel_size == 'small':
                    basins[development_scenario][tmp_basin['Basin_ID']].DrainageDischarge = small_channel
                elif channel_size == 'mid':
                    basins[development_scenario][tmp_basin['Basin_ID']].DrainageDischarge = mid_channel
                elif channel_size == 'large':
                    basins[development_scenario][tmp_basin['Basin_ID']].DrainageDischarge = large_channel
                    basins[development_scenario][tmp_basin['Basin_ID']].ExitChannel = True
                else:
                    print('wrong channel size inserted', channel_size)
            elif tmp_basin['Drainage channel'] == 'no':
                basins[development_scenario][tmp_basin['Basin_ID']].DrainageChannel = False
                drainage = tmp_basin['other drainage ']
                if drainage == 'low':
                    basins[development_scenario][tmp_basin['Basin_ID']].DrainageDischarge = low_drain
                elif drainage == 'mid':
                    basins[development_scenario][tmp_basin['Basin_ID']].DrainageDischarge = mid_drain
                elif drainage == 'high':
                    basins[development_scenario][tmp_basin['Basin_ID']].DrainageDischarge = high_drain
                else:
                    print('wrong drainage capacity chosen', drainage)
            else:
                print('wrong drainage channel input', tmp_basin['Drainage channel'])

            try:
                basins[development_scenario][tmp_basin['Basin_ID']].ToBasinRelativeElevation = float(
                    tmp_basin['To_basin_diff'])
            except ValueError:
                if tmp_basin['To_basin_diff'] == 'sea':
                    basins[development_scenario][tmp_basin['Basin_ID']].ToBasinRelativeElevation = tmp_basin['To_basin_diff']
                    basins[development_scenario][tmp_basin['Basin_ID']].OutletElevation = tmp_basin['Outlet_elevation']
                else:
                    print('wrong basin_diff input', tmp_basin['To_basin_diff'])


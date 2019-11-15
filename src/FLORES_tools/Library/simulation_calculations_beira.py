# -*- coding: utf-8 -*-
"""
This file is part of the simulation of the MODOS-model. The MODOS-model can be used to compare various flood risk
reduction strategies and consists of a Simulation part and an Evaluation part. the Simulation requires four .py files:

- MODOS_simulation_model_<Project>_<Version>.py
    - simulation_definitions_<Project>_<Version>.py
    - simulation_data_<Project>_<Version>.py
    - simulation_calculations_<Project>_<Version>.py

This file lists functions used in the MODOS simulation. these functions are used in the main 'MODOS_simulation_model'
and the 'simulation_definitions' pages.

Project: Beira, Mozambique
Last updated: 2018-08-24(Erik)

Author: E.C. van Berchum(TU Delft)

"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import scipy.stats as ss
import math
import numpy as np
import warnings
from scipy import interpolate


def run_hydraulic_calculation(region_layout, hydraulic, strategy):
    """

    :param region_layout:
    :param hydraulic:
    :param strategy:
    :return:
    """
    max_t = hydraulic.StormDuration + 1
    time = np.arange(0, max_t+0.1, 0.1) * 3600
    timestep = int(time[1] - time[0])
    surgelist = pad_or_truncate(list(hydraulic.SurgeSeries), len(time))
    h_close = 1.75                                            # height at which they close the barriers
    time_fail = 145                                           # surgelist.index(max(surgelist))
    time_fail2 = surgelist.index(max(surgelist)) - 10
    old_basin_volumes = {}
    flow_interbasin = {0: {}}
    outflow_drainage = {}
    inflow_rain = {}
    inflow_storm = {}
    outflow_drain = {}
    outflow_infiltration = {}
    rain_series = pad_or_truncate(hydraulic.RainSeries, len(time))
    total_volume_in_system = {}
    retention = {}
    volume_retention = {}
    listbasin = [basin for basin in region_layout.Basins if basin is not 0]
    for basin in region_layout.Basins:
        retention[basin] = {}
        flow_interbasin[basin] = {}
        old_basin_volumes[basin] = {}
        outflow_drainage[basin] = {}
        for scenario in strategy.ListScenarios:
            retention[basin][scenario] = [0] * (len(time) + 1)
            flow_interbasin[0][scenario] = [0] * (len(time) + 1)     # added so the drain_to_basin does not give an error. if basin does not drain on other basin, basin 0 is chosen. flow should always be 0
            flow_interbasin[basin][scenario] = [0] * (len(time) + 1)
            region_layout.Basins[basin].ScenarioWaterLevels[scenario] = [0] * len(time)
            old_basin_volumes[basin][scenario] = 0
            outflow_drainage[basin][scenario] = [0] * len(time)
            inflow_rain[scenario] = 0
            inflow_storm[scenario] = 0
            outflow_drain[scenario] = 0
            volume_retention[scenario] = 0
            outflow_infiltration[scenario] = 0
            total_volume_in_system[scenario] = [0] * len(time)

    for i in range(len(time)):
        for basin in listbasin:
            tmp_basin_volume_change = {}
            absolute_surrounding_water_levels = {}
            for scenario in strategy.ListScenarios:
                drain_to_other_basin = {}
                tmp_basin_volume_change[scenario] = flow_interbasin[basin][scenario][i]   # first: flow from other basins
                absolute_surrounding_water_levels[scenario] = region_layout.Basins[basin].get_absolute_surrounding_water_levels(region_layout.Basins, scenario, i-1)
                tmp_inflow_rain = region_layout.Basins[basin].run_rain_module(rain_series[i], timestep)  # second: inflow by rain
                tmp_outflow_infiltration = region_layout.Basins[basin].calculate_infiltration(timestep, region_layout.Basins[basin].ScenarioWaterLevels[scenario][i-1])           # third: outflow by infiltration

                tmp_basin_volume_change[scenario] += (tmp_inflow_rain - tmp_outflow_infiltration)
                inflow_rain[scenario] += tmp_inflow_rain
                outflow_infiltration[scenario] += tmp_outflow_infiltration
                tmp_inflow_storm = 0
                for sequence in region_layout.Layers:
                    if region_layout.Layers[sequence].Type == 'Line of Defense':
                        for location in region_layout.Layers[sequence].FDLocations:
                            if basin in region_layout.Layers[sequence].FDLocations[location].BasinCodesWidths:
                                used_measure = region_layout.Basins[basin].UsedMeasure
                                if region_layout.Layers[sequence].FDLocations[location].IncomingBasin == 0:
                                    outside_water_level = surgelist[i]
                                    waveheight = min(hydraulic.WaveHeight, 0.5 * (max(surgelist[i] - hydraulic.MeanSeaLevel, 0)))
                                else:
                                    incoming_basin = region_layout.Layers[sequence].FDLocations[location].IncomingBasin
                                    outside_water_level = region_layout.Basins[incoming_basin].ScenarioWaterLevels[scenario][i-1] + region_layout.Basins[incoming_basin].Contours[0].MinHeight
                                    waveheight = 0.5 * (outside_water_level - region_layout.Basins[incoming_basin].Contours[0].MinHeight)
                                inside_water_level_absolute = region_layout.Basins[basin].ScenarioWaterLevels[scenario][i-1] + region_layout.Basins[basin].Contours[0].MinHeight
                                [V_hold, V_fail] = region_layout.Basins[basin].run_storm_surge_module(inside_water_level_absolute, region_layout.Layers[sequence].FDLocations[location], used_measure, outside_water_level, waveheight, time_fail, time_fail2, h_close, i, timestep)  # Fourth: inflow by storm surge
                                if region_layout.Layers[sequence].Scenarios[scenario].Situation[region_layout.Layers[sequence].Name][location] == 'hold':
                                    inflow_storm[scenario] += V_hold
                                    tmp_basin_volume_change[scenario] += V_hold
                                    tmp_inflow_storm = V_hold
                                elif region_layout.Layers[sequence].Scenarios[scenario].Situation[region_layout.Layers[sequence].Name][location] == 'fail':
                                    inflow_storm[scenario] += V_fail
                                    tmp_basin_volume_change[scenario] += V_fail
                                    tmp_inflow_storm = V_fail
                                else:
                                    print('wrong scenario input', scenario, basin, sequence, location)
                                break

                tmp_basin_volume_before_drain = max(old_basin_volumes[basin][scenario] + tmp_basin_volume_change[scenario], 0)
                tmp_volume_into_retention = min(tmp_basin_volume_before_drain, region_layout.Basins[basin].RetentionCapacity - retention[basin][scenario][i])
                retention[basin][scenario][i+1] = retention[basin][scenario][i] + tmp_volume_into_retention  # volume left in retention in the basin
                tmp_basin_volume_before_drain -= tmp_volume_into_retention
                [drain_to_basin, tmp_drain_drop_off] = region_layout.Basins[basin].get_drain_drop_off(surgelist[i-1], region_layout.Basins, scenario, i-1)
                [outflow_drainage[basin][scenario][i], tmp_basin_volume_after_drain, drain_to_other_basin[drain_to_basin], retention[basin][scenario][i+1]] = calculation_drainage(region_layout.Basins[basin], drain_to_basin, tmp_basin_volume_before_drain, timestep, tmp_drain_drop_off, retention[basin][scenario][i+1])   # Fifth: outflow by drainage
                volume_retention[scenario] += (retention[basin][scenario][i+1] - retention[basin][scenario][i])    # total volume for the scenario (basin sum)
                outflow_drain[scenario] += outflow_drainage[basin][scenario][i]
                tmp_absolute_basin_water_level = float(region_layout.Basins[basin].VolumeToHeight(tmp_basin_volume_after_drain))
                [tmp_absolute_basin_water_level, flow_to_other_basins] = region_layout.Basins[basin].get_interbasin_flow(tmp_absolute_basin_water_level, absolute_surrounding_water_levels[scenario])  # Sixth: outflow to other basins
                old_basin_volumes[basin][scenario] = float(region_layout.Basins[basin].HeightToVolume((tmp_absolute_basin_water_level)))
                for to_basin in flow_to_other_basins:
                    flow_interbasin[to_basin][scenario][i+1] += flow_to_other_basins[to_basin]  # outflow in this time step is inflow in other basins in next time step
                flow_interbasin[drain_to_basin][scenario][i+1] += drain_to_other_basin[drain_to_basin]

                region_layout.Basins[basin].ScenarioWaterLevels[scenario][i] = float(tmp_absolute_basin_water_level - region_layout.Basins[basin].Contours[0].MinHeight)   # VolumeToHeight gives absolute height, not inundation
                total_volume_in_system[scenario][i] += old_basin_volumes[basin][scenario]
            for scenario in ['1.1']:
                tmp_total_in = tmp_inflow_rain + tmp_inflow_storm + flow_interbasin[basin][scenario][i]
                tmp_total_out = tmp_outflow_infiltration + outflow_drainage[basin][scenario][i] + tmp_volume_into_retention + sum(flow_to_other_basins.values())  # + old_basin_volumes[basin][scenario]
                # print("Timestep = {}. Basin = {}. scenario = {}, inundation = {}, old_height = {}, volume_change = {}, flow_into_basin = {}, BorderHeights = {}".format(i, basin, scenario, round(region_layout.Basins[basin].ScenarioWaterLevels[scenario][i], 4), round(region_layout.Basins[basin].ScenarioWaterLevels[scenario][i] + region_layout.Basins[basin].Contours[0].MinHeight,4), round(tmp_basin_volume_after_drain, 4), round(flow_interbasin[basin][scenario][i],4), region_layout.Basins[basin].BorderHeights))
                # print('In: rain: {}, storm, {}, flow in: {}. Out: infiltration: {}, drain: {}, retention: {},
            # flow_away: {} volume_left: {}. total in: {}, total out {}, difference: {}'.format(tmp_inflow_rain,
            # tmp_inflow_storm, flow_interbasin[basin][scenario][i], tmp_outflow_infiltration,
            # outflow_drainage[basin][scenario][i],tmp_volume_into_retention, sum(flow_to_other_basins.values()) +
            # drain_to_other_basin[drain_to_basin], old_basin_volumes[basin][scenario], tmp_total_in, tmp_total_out,
            # tmp_total_in - tmp_total_out))

    for basin in region_layout.Basins:
        for scenario in strategy.ListScenarios:
            region_layout.Basins[basin].get_maximum_water_level(scenario)

    return [inflow_rain, inflow_storm, outflow_drain, outflow_infiltration, volume_retention]


def calculations_storm_surge_module(basin, waterlevel_before, location, measure, outside_waterlevel, waveheight,
                                    time_fail_land, time_fail_water, h_close, time, timestep):
    """
    This function simulates storm surge hitting a Line of Defense. For 1 timestep, it calculates the volume of water
    passing the Line of Defense. The volume is the total amount of volume flowing into 1 basin. On the Line of Defense,
    a flood defense can be placed. The volume is calculated both for the situation where it holds and fails.

    :param basin: The drainage basin that receives the volume of water
    :param waterlevel_before: Absolute water level in the basin at the start of the timestep [m+MSL]
    :param location: Part of Line of Defense where the storm surge hits. can be Land (e.g. Coast) or Water (e.g. River)
    :param measure: Flood defense that can be placed on the location.
    :param outside_waterlevel: Water level on the outer side of the Line of Defense at the time of timestep. [m+MSL]
    :param waveheight: Height of waves at the location. breaking should already have been taken into account[m]
    :param time_fail_land: Moment in time where a land-based measure is modelled to fail
    :param time_fail_water: Moment in time where a water-based measure is modelled to fail
    :param h_close: Moment in them where a water-based barrier is modelled to close
    :param time: Current moment in time, starting from the start of the storm
    :param timestep: length of a time step [s]

    Important variables:
    Q_hold: Discharge of water entering the basin per second in case a measure holds [m^3/s]
    Q_fail: Discharge of water entering the basin per second in case a measure fails [m^3/s]

    :return V_hold: Volume of water entering the basin during the timestep in case a measure holds [m^3]
    :return V_fail: Volume of water entering the basin during the timestep in case a measure fails [m^3]
    """

    Q_hold = None
    Q_open = None

    if location.Type == 'Land':
        if waterlevel_before > location.Height + 1:
            Q_open = formula_broad_weir(location, outside_waterlevel, waterlevel_before) * basin.Width  # [m3/s]
        else:
            Q_open = location.get_overtopping(outside_waterlevel, location.Height, waveheight) * basin.Width

        if measure:
            if waterlevel_before > measure.Height + 1:  # bay level is higher than barrier, so it acts as broad weir
                Q_hold = formula_broad_weir(measure, outside_waterlevel, waterlevel_before) * basin.Width
            else:  # bay level is lower than the barrier, so it overtops/overflows
                Q_hold = measure.get_overtopping(outside_waterlevel, measure.Height, waveheight) * basin.Width

            if time < time_fail_land:
                Q_open = Q_hold
            else:
                Q_open = Q_open * measure.BreachFactor + Q_hold * (1 - measure.BreachFactor)
        else:
            Q_hold = Q_open

    elif location.Type == 'Water':
        Q_open = calculate_open_channel_flow(location, outside_waterlevel, waterlevel_before)

        if measure and outside_waterlevel > h_close:
            if waterlevel_before > measure.Height + 1:  # Acts like a weir in this situation
                Q_hold = formula_broad_weir(location, outside_waterlevel,
                                            waterlevel_before) * basin.Width
            else:  # bay level is lower than surge and lower than barrier, so it overtops/overflows
                Q_hold = measure.get_overtopping(outside_waterlevel, measure.Height,
                                                 waveheight) * basin.Width
            if time < time_fail_water:
                Q_open = Q_hold
        else:
            Q_hold = Q_open

    V_hold = Q_hold * timestep
    V_open = Q_open * timestep

    return [V_hold, V_open]


def calculate_overtopping(self, waterlevel, h_barrier, Hs):
    """
    This function calculates discharge past a barrier (measure/location) in case of overtopping or overflow.

    Calculations are based on Overtopping Manual(2016)

    :param self: The barrier that the water has to pass
    :param waterlevel:  Water level on the outer side at the time of timestep. [m+MSL]
    :param h_barrier: height of barrier [m+MSL]
    :param Hs: significant wave height [m]

    Important variables:
    Rc: freeboard [m]

    :return q: discharge past a barrier [m^3/m/s]
    """
    warnings.filterwarnings("error")
    Rc = h_barrier - waterlevel
    if Rc > 2.5:
        return 0.0

    if self.Slope < 0.5:
        if Rc > 0:
            try:
                y1 = 0.026/math.sqrt(self.Slope)*self.Irribaren*(
                                    math.exp(-((2.5*Rc/(self.Irribaren*Hs))**1.3))*math.sqrt(9.81*Hs**3))
                y2 = 0.1035 * math.exp(-((1.35 * Rc / Hs) ** 1.3)) * math.sqrt(9.81 * Hs ** 3)
                q = min(y1, y2)
            except RuntimeWarning:
                q = 0
        else:
            q = 0.54*math.sqrt(9.81*(abs((-Rc)**3)))+0.026/math.sqrt(self.Slope)*(
                    self.Irribaren*math.sqrt(9.81*Hs**3))
    else:
        if Rc > 0:
            try:
                q = 0.047 * math.exp(-((2.35 * Rc / Hs) ** 1.3)) * math.sqrt(9.81 * Hs ** 3)
            except RuntimeWarning:
                q = 0
        else:
            q = 0.54*math.sqrt(9.81*(abs((-Rc)**3)))+0.047*math.sqrt(9.81*Hs**3)

    return q


def calculate_open_channel_flow(channel, water_level_outside, water_level_inside):
    """
    This functions calculates discharge past a barrier in case it is connected through an open channel

    :param channel: connecting body of water between outside water level and inside water level
    :param water_level_outside: absolute height of water level on the outer (sea) side of the channel [m+MSL}
    :param water_level_inside: absolute height of water level on the inner side of the channel [m+MSL]

    :return q: discharge through the channel [m^3/m/s]
    """

    d_water = channel.Depth + (water_level_outside + water_level_inside) / 2
    diff = water_level_outside - water_level_inside
    q = 0
    if diff != 0 and d_water > 0:
        sign = diff / abs(diff)
        surf_c = channel.Width * d_water
        radius = surf_c / (2 * d_water + channel.Width)          # de Vries
        factor = 0.5
        tmp = abs(diff) / (factor + 10/(channel.Chezy**2) * channel.Length / radius) * 10 * surf_c**2
        q = sign * math.sqrt(tmp)
    return q


def formula_broad_weir(land, water_level_outside, water_level_inside):
    """
    Calculates discharge past a barrier (measure/ location) in case of a broad weir.

    more information on : http://cirpwiki.info/wiki/Weirs
    :param land: the barrier that the water has to pass
    :param water_level_outside: absolute height of water level on the outer (sea) side of the channel [m+MSL}
    :param water_level_inside: absolute height of water level on the inner side of the channel [m+MSL]

    :return q: discharge past the barrier [m^3/m/s]
    """

    q = 0
    h_upper = max(water_level_outside, water_level_inside)
    h_lower = min(water_level_outside, water_level_inside)
    
    if land.Type == 'Water':
        height = 0
    else:
        height = land.Height
        
    if h_upper > height:
        C_w = 0.55        # HEC2010 predicts 0.46-0.55 for broad-crested weirs
        signum = 1
        if h_lower/h_upper <= 0.67:
            C_df = 1
        else:
            C_df = 1-27.8*(h_lower/h_upper-0.67)**3
        
        if water_level_inside > water_level_outside:
            signum = -1  # water flows outwards
        q = signum * C_df * C_w * math.sqrt(9.81) * (h_upper-height)**(3/2)
        
    return q


def calculations_rain_module(basin, rain_intensity, timestep):
    """
    Calculates the volume of rain falling on a basin during 1 timestep.

    :param basin: Basin in which the rain is falling
    :param rain_intensity: amount of rain falling per hour [mm/hour]
    :param timestep: amount of seconds per timestep

    :return inflow_volume_rain:  volume of rain entering the basin in 1 timestep [m^3]
    """

    inflow_volume_rain = basin.SurfaceArea * (rain_intensity / 1000) * (timestep/3600)
    return inflow_volume_rain


def calculate_max_series(series):
    """
    Calculate the maximum of a series

    :param series: list of values where we want to know the maximum of.

    :return max(series): highest value in the series
    """

    assert type(series) is list and len(series) != 0
    return max(series)


def calculate_cost_construction_repair(structure, location):
    """
    Calculates the construction costs and repair costs of building a structure. It is assumed that the construction
    costs money to start with and it costs aditional money to make the structure higher. The repair cost is assumed
    to be the costs to replace the parts of the structure that were destroyed (breached).

    :param structure: The structure that we want to know the costs of
    :param location: the location where the structure is built

    :return [construction_cost, repair cost]: the total construction and repair cost of the structure
    """

    if location.Type == 'Water':
        height = 0
    else:
        height = location.Height
            
    construction_cost = (structure.CostConstant+structure.CostVariable*(structure.Height-height))*location.Width / 1000
    repair_cost = structure.ReplacementFactor * construction_cost * structure.BreachFactor
    
    return [construction_cost, repair_cost]


def calculate_failure_probability(measure, water_level, waveheight):
    """
    Calculates the failure probability of a structural flood risk reduction measure. The fragility curve is now
    schematized as a cumulative normal distribution.

    :param measure: structure
    :param water_level: maximum load against the structure [m+MSL]
    :param waveheight: maximum wave height loading the structure [m]

    :return fail_probability: probability of failure of the structure, between 0 and 1
    """
    waveheight += 0.01  # to prevent it from being 0, that would crash the following line
    fail_probability = ss.norm.cdf(water_level, loc=measure.HeightFail-waveheight/3, scale=0.25*waveheight)
    return fail_probability
    

def calculate_wave_height(outside_wave_height, max_surge):
    """
    Calculates the wave height. Waves are assumed to break at 0.45 the water level

    :param outside_wave_height: maximum wave height during event far of the coast [m]
    :param max_surge: maximum water level at the location where the wave height is calculated [m+MSL]

    :return wave_height: wave height at the location [m]
    """
    wave_height = min(outside_wave_height, 0.45*max_surge)
    return wave_height


def create_surge_series(max_surge, storm_duration, amplitude, timestep_model, mean_sea_level):
    """
    This function makes a time series of the surge, consisting of tide and a surge. The surge is schematized as a
    half sine. the tide is schematized to have its peak halfway through the storm duration (when surge is max)

    :param max_surge: the maximum additional surge of the water level because of the storm [m]
    :param storm_duration: length of the storm [hours]
    :param amplitude: normal amplitude of the tide [m]
    :param timestep_model: length of one timestep [s]
    :param mean_sea_level: absolute height of the mean sea level [m]

    :return: time series of water levels [m+MSL]
    """

    TIDE_INTERVAL = 12.417  # 12 hours and 25 minutes
    time1 = np.linspace(0, storm_duration, int(storm_duration/timestep_model + 1), endpoint=True)
    tide = amplitude * np.cos(np.pi / (TIDE_INTERVAL / 2) * time1 - 0.5 * storm_duration) + mean_sea_level
    surge = max_surge * np.sin(np.pi / storm_duration * time1) ** 5
    total_storm_surge_series = tide + surge

    return list(total_storm_surge_series)


def calculate_interpolate(old_series, old_x, new_x):
    """
    This function interpolates a function (old_x,old_series) and projects it on a new x (new_x). the result is a series
    of (new_x, new_series). This can be useful to change a timeseries into different lengths.

    :param old_series: The y-values of the function that needs to be changed
    :param old_x: the x-values of the function that needs to be changed
    :param new_x: the x-values of the new function

    :return: the y-values of the new function
    """

    series_tmp = interpolate.interp1d(old_x, old_series, kind='cubic')
    new_series = list(series_tmp(new_x))

    return new_series


def tolist(input_string, splitby, export_type=int):
    """
    This function is used to divide list that has been imported as a string into an actual list. For example, if the
    import is '1;2;3;4', the output is [1,2,3,4]

    :param input_string: string value, consisting of divided numbers
    :param splitby: punctuation mark used to divide the numbers
    'param expert_type: type of the variables in the final list. standard = integer

    :return: list of values
    """

    if type(input_string) == str:
        return list(map(export_type, input_string.split(splitby)))
    else:
        str_variable = str(int(input_string))
        return list(map(export_type, str_variable.split(splitby)))


def calculation_drainage(basin, drain_to_basin, volume, timestep, drain_drop_off, retention):
    """
    This function calculates the volume that is drained from a drainage basin, as well as the remaining volume. The
    water can drain to another basin (flow_interbasin) or out to sea/sewerage (outflow_drainage). The water level in the
    drainage basin is compared with the water level of the receiving body (basin/sea). If the receiving water level is
    higher, no water flows. In other cases, a volume of water drains from the basin to the receiving basin. This volume
    is less when the difference in water level is small (<1 m).


    :param basin: The drainage basin that drains
    :param drain_to_basin: the number of the drainage basin that receives the water. '0' represents sea/sewerage/none
    :param volume: The total volume of water inside the basin at the start of the calculation [m^3]
    :param timestep: length of time of a model timestep [s]
    :param drain_to_water_level: Water level in the drainage basin that receives the drained volume of water [m+MSL]
    :param basin_water_level: Water level in the drainage basin that drains [m+MSL]

    :return: List with three entries:
    :return outflow_drainage: volume of water that drains out of the system [m^3]
    :return remaining_volume: volume of water that remains in the drainage basin [m^3]
    :return flow_interbasin: volume of water that flow to another drainage basin [m^3]
    """

    drain_factor = 1  # if drain_drop_off is more than 1, nothing happens in following statement and factor = 1
    if drain_drop_off < 0:
        return [0, volume, 0, retention]
    elif drain_drop_off < 1.0:
        drain_factor = min(0.25 + drain_drop_off*3/4, 1)

    max_drainage_capacity = basin.DrainageDischarge * timestep * drain_factor
    outflow_drainage = min(volume + retention, max_drainage_capacity)
    remaining_volume = volume - outflow_drainage
    if remaining_volume < 0:
        retention += remaining_volume   # negative remaining volume, so retention decreases
        remaining_volume = 0
    flow_interbasin = 0
    if drain_to_basin != 0:   # if water flows to another basin ( 0 represents 'sea' or 'none')
        flow_interbasin = outflow_drainage
        outflow_drainage = 0

    return [outflow_drainage, remaining_volume, flow_interbasin, retention]


def pad_or_truncate(some_list, target_len):
    """
    This function shortens or extends a list. When it is extended, extra 0 values are added. This can be helpful to
    simulate what happens after a storm has passed.

    :param some_list: The original list of values
    :param target_len: the wanted length of the list

    :return: a list of values with length of target_len
    """
    return some_list[:target_len] + [0]*(target_len - len(some_list))

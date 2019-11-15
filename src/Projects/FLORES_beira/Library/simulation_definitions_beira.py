""""
Top level objects:
Region layout
    Basins
    Lines of Defense

Strategy
    Structural Measures
        Levee
        Seawall
        Storm surge barrier
    Nature-based Solutions
    Disaster Management

Hydraulic Boundary Conditions
    Surge
    Rain

"""
from simulation_calculations_beira import calculations_storm_surge_module, calculate_max_series, \
                                              create_surge_series, calculate_overtopping, \
                                              calculate_cost_construction_repair, calculate_failure_probability, \
                                              calculate_interpolate, calculations_rain_module
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy import interpolate


class RegionLayout(object):
    """"
    The Region layout consists of all information on the region itself.
    """

    def __init__(self, basindict, layersdict):

        self.Basins = basindict
        self.Layers = layersdict
        self.Damages = {}

    def plot_4_basin_water_levels(self, basin1, basin2, basin3, basin4, strategy):
        """

        :param basin1:
        :param basin2:
        :param basin3:
        :param basin4:
        :param strategy:
        :return:
        """

        x = [i/10 for i in range(len(self.Basins[basin1].ScenarioWaterLevels['1.1']))]
        y = {}
        y2 = {}
        plt.style.use('seaborn-darkgrid')
        # palette = plt.get_cmap('Set1')
        num = 0
        for basin in [basin1, basin2, basin3, basin4]:
            num += 1
            plt.subplot(2, 2, num)
            for scenario in strategy.ListScenarios:
                y[scenario] = self.Basins[basin].ScenarioWaterLevels[scenario]
                plt.plot(x, y[scenario], marker='')
            for border in self.Basins[basin].BorderHeights:
                y2[border] = [self.Basins[basin].BorderHeights[border]
                              - self.Basins[basin].Contours[0].MinHeight] * len(x)
                plt.plot(x, y2[border], marker='')

            plt.xlim(0, 30)
            plt.ylim(0, 6)

            if num in range(1):
                plt.tick_params(labelbottom=False)
            if num in [1, 3]:
                plt.tick_params(labelleft=False)

            plt.title('Basin:' + str(basin), loc='left')
        plt.suptitle('Inundation levels for all scenarios in 4 basins')

    def show_results(self, basin1, basin2, basin3, basin4, basin5, basin6, basin7, basin8, basin9, strategy,
                     return_period_storm, return_period_rain, struc_measure_east, struc_measure_west,
                     struc_measure_inland, drainage_measure_1, drainage_measure_2, retention_measure_1, retention_measure_2,
                     emergency_measure_1, emergency_measure_2, h_struc_measure_east, h_struc_measure_west,
                     h_struc_measure_inland):

        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(331)
        ax2 = fig.add_subplot(332)
        ax3 = fig.add_subplot(333)
        ax4 = fig.add_subplot(334)
        ax5 = fig.add_subplot(335)
        ax6 = fig.add_subplot(336)
        ax7 = fig.add_subplot(337)
        ax8 = fig.add_subplot(338)
        ax9 = fig.add_subplot(339)

        x = [i / 10 for i in range(len(self.Basins[basin1].ScenarioWaterLevels['1.1']))]
        y = {basin1: {}, basin2: {}, basin3: {}, basin4: {}, basin5: {}, basin6: {}, basin7: {}, basin8: {}, basin9: {}}
        for scenario in strategy.ListScenarios:
            y[basin1][scenario] = self.Basins[basin1].ScenarioWaterLevels[scenario]
            ax1.set(title='Basin:' + str(basin1))
            ax1.plot(x, y[basin1][scenario])
            y[basin2][scenario] = self.Basins[basin2].ScenarioWaterLevels[scenario]
            ax2.set(title='Basin:' + str(basin2))
            ax2.plot(x, y[basin2][scenario])
            y[basin3][scenario] = self.Basins[basin3].ScenarioWaterLevels[scenario]
            ax3.set(title='Basin:' + str(basin3))
            ax3.plot(x, y[basin3][scenario])
            y[basin4][scenario] = self.Basins[basin4].ScenarioWaterLevels[scenario]
            ax4.set(title='Basin:' + str(basin4))
            ax4.plot(x, y[basin4][scenario])
            y[basin5][scenario] = self.Basins[basin5].ScenarioWaterLevels[scenario]
            ax5.set(title='Basin:' + str(basin5))
            ax5.plot(x, y[basin5][scenario])
            y[basin6][scenario] = self.Basins[basin6].ScenarioWaterLevels[scenario]
            ax6.set(title='Basin:' + str(basin6))
            ax6.plot(x, y[basin6][scenario])
            y[basin7][scenario] = self.Basins[basin7].ScenarioWaterLevels[scenario]
            ax7.set(title='Basin:' + str(basin7))
            ax7.plot(x, y[basin7][scenario])
            y[basin8][scenario] = self.Basins[basin8].ScenarioWaterLevels[scenario]
            ax8.set(title='Basin:' + str(basin8))
            ax8.plot(x, y[basin8][scenario])
            y[basin9][scenario] = self.Basins[basin9].ScenarioWaterLevels[scenario]
            ax9.set(title='Basin:' + str(basin9))
            ax9.plot(x, y[basin9][scenario])

        plt.suptitle('Main results Beira simulation\n'
                     'Return period storm: {} year, rain {} year\n'
                     'Structural measures: East: {} ({} m height), West: {} ({} m height), Inland: {} ({} m high)\n'
                     'Drainage measure: {}'.format(return_period_storm, return_period_rain, struc_measure_east,
                                                   h_struc_measure_east, struc_measure_west, h_struc_measure_west,
                                                   struc_measure_inland, h_struc_measure_inland, drainage_measure_1))

        plt.show()

    def __str__(self):

        return ('    Region Layout\n'
                '   Basin dictionary:    {Basins}\n'
                '   Layer dictionary:    {Layers}\n'
                ).format(**self.__dict__)


class FRRStrategy(object):

    def __init__(self, chosen_measures, all_measures):

        self.ChosenMeasures = chosen_measures
        self.AllMeasures = all_measures
        self.StructuralMeasures = {}
        self.NatureBasedSolutions = {}
        self.EmergencyMeasures = {}
        self.DrainageMeasures = {}
        self.RetentionMeasures = {}
        self.ScenarioProbabilities = {}
        self.ListScenarios = []
        self.TotalExpectedRepair = None
        self.TotalConstructionCost = 0

    def get_measures(self, structural_measures_height, layers):
        """"
        Load data on the measures into the strategy dictionary
        """
        self.ChosenMeasures = [x for x in self.ChosenMeasures if x != 'none']
        for measure in self.ChosenMeasures:
            if not measure:
                print('wrong input, use "none" instead of None', measure)
            if self.AllMeasures[measure].Type == 'Structural':
                if self.AllMeasures[measure].Layer == 'Coast' and self.AllMeasures[measure].Location == 'East':
                    self.AllMeasures[measure].Height = float(structural_measures_height[0])
                    self.AllMeasures[measure].HeightFail = self.AllMeasures[measure].get_height_fail()
                    [self.AllMeasures[measure].ConstructionCost, self.AllMeasures[measure].RepairCost] = \
                     self.AllMeasures[measure].get_cost_construction_repair(layers[1].FDLocations['East'])
                elif self.AllMeasures[measure].Layer == 'Coast' and self.AllMeasures[measure].Location == 'West':
                    self.AllMeasures[measure].Height = float(structural_measures_height[1])
                    self.AllMeasures[measure].HeightFail = self.AllMeasures[measure].get_height_fail()
                    [self.AllMeasures[measure].ConstructionCost, self.AllMeasures[measure].RepairCost] = \
                     self.AllMeasures[measure].get_cost_construction_repair(layers[1].FDLocations['West'])
                elif self.AllMeasures[measure].Layer == 'Inland':
                    self.AllMeasures[measure].Height = float(structural_measures_height[2])
                    self.AllMeasures[measure].HeightFail = self.AllMeasures[measure].get_height_fail()
                    [self.AllMeasures[measure].ConstructionCost, self.AllMeasures[measure].RepairCost] = \
                     self.AllMeasures[measure].get_cost_construction_repair(layers[3].FDLocations['Inland road'])
                self.StructuralMeasures[measure] = self.AllMeasures[measure]

            elif self.AllMeasures[measure].Type == 'Nature based Solution':
                self.NatureBasedSolutions[measure] = self.AllMeasures[measure]
            elif self.AllMeasures[measure].Type == 'Drainage':
                self.DrainageMeasures[measure] = self.AllMeasures[measure]
            elif self.AllMeasures[measure].Type == 'Emergency':
                self.EmergencyMeasures[measure] = self.AllMeasures[measure]
            elif self.AllMeasures[measure].Type == 'Retention':
                self.RetentionMeasures[measure] = self.AllMeasures[measure]
            else:
                print("wrong type input for measure: ", measure)

    def get_list_scenarios(self, region_layout):

        self.ListScenarios = list(region_layout.Layers[3].Scenarios.keys())

    def get_probabilities(self, region_layout, hydraulic):
        Pf = {}
        repair_costs = {}
        for sequence in region_layout.Layers:
            if region_layout.Layers[sequence].Type == 'Line of Defense':
                for location in region_layout.Layers[sequence].FDLocations:
                    Pf[location] = {}
                    repair_costs[location] = {}
                    for scenario in ['1.1', '2.1', '3.1', '4.1']:
                        if region_layout.Layers[sequence].FDLocations[location].UsedMeasure:
                            repair_costs[location] = self.StructuralMeasures[region_layout.Layers[sequence].FDLocations[location].UsedMeasure].RepairCost
                            if region_layout.Layers[sequence].Name == 'Inland':
                                max_inundation = region_layout.Basins[region_layout.Layers[sequence].FDLocations[location].IncomingBasin].ScenarioMaxWaterLevels[scenario]
                                ground_elevation = region_layout.Basins[region_layout.Layers[sequence].FDLocations[location].IncomingBasin].Contours[0].MinHeight
                                Pf[location][scenario] = self.StructuralMeasures[region_layout.Layers[sequence].FDLocations[location].UsedMeasure].get_failure_probability(max_inundation + ground_elevation, 0.5 * max_inundation)
                            else:
                                water_level = max(hydraulic.SurgeSeries)
                                Pf[location][scenario] = self.StructuralMeasures[region_layout.Layers[sequence].FDLocations[location].UsedMeasure].get_failure_probability(water_level, min(hydraulic.WaveHeight, 0.5 * water_level))
                        else:
                            Pf[location][scenario] = 0
                            repair_costs[location] = 0

        probabilities = [(1-Pf['East']['1.1'])*(1-Pf['West']['1.1'])*(1-Pf['Inland road']['1.1']),
                         (1 - Pf['East']['1.1']) * (1 - Pf['West']['1.1']) * Pf['Inland road']['1.1'],
                         Pf['East']['2.1'] * (1 - Pf['West']['2.1']) * (1 - Pf['Inland road']['2.1']),
                         Pf['East']['2.1'] * (1 - Pf['West']['2.1']) * Pf['Inland road']['2.1'],
                         (1 - Pf['East']['3.1']) * Pf['West']['3.1'] * (1 - Pf['Inland road']['3.1']),
                         (1 - Pf['East']['3.1']) * Pf['West']['3.1'] * Pf['Inland road']['3.1'],
                         Pf['East']['4.1'] * Pf['West']['4.1'] * (1 - Pf['Inland road']['4.1']),
                         Pf['East']['4.1'] * Pf['West']['4.1'] * Pf['Inland road']['4.1']]
        repair_scenarios = [0, repair_costs['Inland road'], repair_costs['East'], repair_costs['East'] + repair_costs['Inland road'],
                            repair_costs['West'], repair_costs['West'] + repair_costs['Inland road'], repair_costs['East'] +
                            repair_costs['West'], repair_costs['East'] + repair_costs['West'] + repair_costs['Inland road']]

        self.ScenarioProbabilities = dict(zip(self.ListScenarios, probabilities))
        self.TotalExpectedRepair = sum([a * b for (a, b) in zip(repair_scenarios, probabilities)])

    def get_construction_costs(self):

        for measure in self.ChosenMeasures:
            self.TotalConstructionCost += self.AllMeasures[measure].ConstructionCost
        total_cost = self.TotalConstructionCost + self.TotalExpectedRepair
        return total_cost


class HydraulicBoundaryConditions(object):
    """
    Hydraulic boundary conditions of one simulation

    Top level surge data refers to coastal conditions, surge conditions at inland layers can be added with
    'AddSurgeLayer'-command. top level rain data is applied for all basins. Wind intensity and wave height is related
    to return period of the storm surge.

    :ReturnPeriodStorm      : return period of storm surge in simulation in years, chosen as input
    :ReturnPeriodRain       : return period of rain event in simulation in years, chosen as input
    :SurgeSeries            : time series of storm surge water levels in [m+MSL], loaded from data
    :RainSeries             : time series of rain intensities in [mm], loaded from data
    :WindSeries             : time series of general wind direction according to wind rose (8 dir), loaded from data


    """
    def __init__(self, storm, rain, rain_series, wind_series, max_surge, max_rain, outside_wave_height,
                 wind_velocity, storm_duration, timestep_model, tide_amplitude, msl, surge_climate_scenarios, rain_climate_scenarios):

        self.ReturnPeriodStorm = storm
        self.ReturnPeriodRain = rain
        self.SurgeSeries = create_surge_series(max_surge, storm_duration, tide_amplitude, timestep_model, msl)
        self.RainSeries = rain_series
        self.WindSeries = wind_series
        self.MaxLevel = max_surge
        self.MaxRainIntensity = max_rain
        self.WindVelocity = wind_velocity
        self.WaveHeight = outside_wave_height
        self.StormDuration = storm_duration
        self.SurgeLayerCount = 1
        self.InlandSurge = {}
        self.TideAmplitude = tide_amplitude
        self.MeanSeaLevel = msl
        self.SurgeClimate = surge_climate_scenarios
        self.RainClimate = rain_climate_scenarios

    def __str__(self):

        return ('    STORM SURGE\n'
                '   Return period storm:    {ReturnPeriodStorm}\n'
                '   Maximum surge level:    {MaxLevel}\n'
                '   Outside wave height:    {WaveHeight}\n'
                '   Storm duration:         {StormDuration}\n'
                '   Maximum wind velocity:  {WindVelocity}\n'
                '   \n'
                '     RAIN\n'
                '   Return period rain:     {ReturnPeriodRain}\n'
                '   Maximum rain intensity: {MaxRainIntensity}\n'
                '\n'
                ''
                ).format(**self.__dict__)

    def apply_climate_scenario(self, climate_scenario):

        if climate_scenario is 'none':
            return
        max_tide = max(self.SurgeSeries)
        self.SurgeSeries = [self.SurgeSeries[i]*((max_tide+self.SurgeClimate[climate_scenario])/max_tide) for i in range(len(self.SurgeSeries))]
        self.RainSeries = [self.RainSeries[j]*((self.MaxRainIntensity * self.RainClimate[climate_scenario])/self.MaxRainIntensity) for j in range(len(self.RainSeries))]
        return


class Basin(object):

    def __init__(self, name, surfacearea=0, contours=[], damfactor=1, width=0):

        self.Name = name
        self.SurfaceArea = surfacearea
        self.Contours = contours
        self.DamageFactor = damfactor
        self.ScenarioDamages = {}
        self.ScenarioExposedPop = {}
        self.BasinExpectedDamage = None
        self.ScenarioProbabilities = {}
        self.AverageElevation = None
        self.ScenarioWaterLevels = {}
        self.ScenarioMaxWaterLevels = {}
        self.Width = width
        self.BorderHeights = {}
        self.SurroundingAreas = {}
        self.InfiltrationRate = None
        self.VolumeToHeight = None
        self.HeightToVolume = None
        self.DrainageChannel = None
        self.DrainageDischarge = None
        self.DrainsToBasin = None
        self.ExitChannel = None
        self.UsedMeasure = None
        self.RetentionCapacity = 0
        self.ToBasinRelativeElevation = None
        self.OutletElevation = None

    def get_expected_basin_damage(self, scenarios):
        """
        Calculates the total expected amount of damage in the basin

        :return:
        """
        self.BasinExpectedDamage = sum(self.ScenarioDamages[scenario]*scenarios.Probabilities[scenario]
                                       for scenario in scenarios)

    def get_maximum_water_level(self, scenario):
        self.ScenarioMaxWaterLevels[scenario] = calculate_max_series(self.ScenarioWaterLevels[scenario])

    def run_storm_surge_module(self, waterlevel_before, location, measure, outside_waterlevel, waveheight, time_fail, time_fail2, h_close, time, timestep):

        [V_hold, V_open] = calculations_storm_surge_module(self, waterlevel_before, location, measure, outside_waterlevel, waveheight, time_fail, time_fail2, h_close, time, timestep)
        return [V_hold, V_open]

    def get_absolute_surrounding_water_levels(self, basins, scenario, i):

        absolute_surrounding_water_levels = {}
        for basin in self.BorderHeights:
            absolute_surrounding_water_levels[basin] = float(basins[basin].ScenarioWaterLevels[scenario][i] + basins[basin].Contours[0].MinHeight)
        return absolute_surrounding_water_levels

    def run_rain_module(self, rain_intensity, timestep):
        inflow_volume_rain = calculations_rain_module(self, rain_intensity, timestep)
        return inflow_volume_rain

    def calculate_infiltration(self, timestep, water_level):
        min_x = self.Contours[0].MinHeight
        max_x = self.Contours[-1].MinHeight
        x = min(self.Contours[-1].MinHeight, water_level + self.Contours[0].MinHeight)
        approx_surface_area = ((x-min_x) / (max_x - min_x)) * self.SurfaceArea      # if there is no water, no infiltration. if basin is full, maximum infiltration
        outflow_volume_infiltration = self.InfiltrationRate / 1000 * approx_surface_area * (timestep/3600)
        return outflow_volume_infiltration

    def get_infiltration_rate(self):

        self.InfiltrationRate = 2.5  # use until we have better method
        return

    def get_volume_inundation_curve(self):

        heights = [self.Contours[0].MinHeight]
        surface_areas = [self.Contours[0].SurfaceArea]
        volumes = [0]
        for contour in range(1, len(self.Contours)):
            heights.append(self.Contours[contour].MinHeight)
            surface_areas.append(round(surface_areas[contour-1] + self.Contours[contour].SurfaceArea, 1))
            volumes.append(volumes[contour-1]+(heights[contour]-heights[contour-1])*surface_areas[contour])
        heights.append(30)
        surface_areas.append(surface_areas[-1])
        volumes.append(volumes[-1]+(heights[-1]-heights[-2])*surface_areas[-1])
        curve_volume_to_height = interpolate.interp1d(volumes, heights, kind='linear')
        curve_height_to_volume = interpolate.interp1d(heights, volumes, kind='linear')
        self.VolumeToHeight = curve_volume_to_height
        self.HeightToVolume = curve_height_to_volume
        return

    def get_interbasin_flow(self, absolute_water_level, surrounding_water_levels):

        flow_to_other_basins = {}
        surrounding_thresholds = {basin: max(surrounding_water_levels[basin], self.BorderHeights[basin]) for basin in
                                  surrounding_water_levels.keys()}
        sorted_thresholds = sorted(surrounding_thresholds.items(), key=lambda kv: kv[1])
        basin_numbers_sorted = []
        surrounding_basins_threshold = []
        surface_areas_sorted = []
        for i in range(len(sorted_thresholds)):
            basin_numbers_sorted.append(sorted_thresholds[i][0])
            surrounding_basins_threshold.append(sorted_thresholds[i][1])
            surface_areas_sorted.append(self.SurroundingAreas[sorted_thresholds[i][0]])
        surrounding_basins_threshold.append(100)

        new_absolute_water_level = absolute_water_level
        for j in range(len(surrounding_basins_threshold)):
            tmp_absolute_water_level = new_absolute_water_level
            if tmp_absolute_water_level > surrounding_basins_threshold[j]:
                area_factor = self.SurfaceArea / surface_areas_sorted[j]
                new_absolute_water_level = (tmp_absolute_water_level * area_factor + surrounding_basins_threshold[j]) / (1 + area_factor)
                flow_to_other_basins[basin_numbers_sorted[j]] = self.HeightToVolume(tmp_absolute_water_level) - self.HeightToVolume(new_absolute_water_level)
            else:
                break

        return [new_absolute_water_level, flow_to_other_basins]

    def get_basin_damage(self, scenario, damage_curves, scenario_probability):
        self.ScenarioDamages[scenario] = 0
        for contour in self.Contours:
            if self.ScenarioMaxWaterLevels[scenario] + self.Contours[0].MinHeight > contour.MinHeight:
                self.ScenarioDamages[scenario] += contour.get_contour_damage(self.ScenarioMaxWaterLevels[scenario], damage_curves)
            else:
                break

        expected_damage = self.ScenarioDamages[scenario] * scenario_probability
        return expected_damage

    def get_exposed_population(self, scenario, scenario_probability):
        self.ScenarioExposedPop[scenario] = 0
        for contour in self.Contours:
            if self.ScenarioMaxWaterLevels[scenario] + self.Contours[0].MinHeight > contour.MinHeight + 0.1: # population is exposed if inundation is more than 0.1 m
                self.ScenarioExposedPop[scenario] += contour.Population
            else:
                break

        expected_exposed_pop = self.ScenarioExposedPop[scenario] * scenario_probability
        return expected_exposed_pop

    def add_drainage_discharge(self, extra):

        self.DrainageDischarge += extra
        return

    def change_basin_population(self, factor):

        for contour in self.Contours:
            contour.Population = contour.Population * factor
        return

    def add_retention_capacity(self, capacity):

        self.RetentionCapacity = capacity
        return

    def __str__(self):

        return ('    BASIN\n'
                '   Name:    {Name}\n'
                '   SurfaceArea:         {SurfaceArea}\n'
                ).format(**self.__dict__)

    def get_drain_drop_off(self, outside_water_level, basins_dict, scenario, j):

        if self.DrainsToBasin == 'sea':
            drain_drop_off = self.ScenarioWaterLevels[scenario][j] + self.OutletElevation - outside_water_level
            drain_to_basin = 0
        else:
            try:
                drain_to_basin = int(self.DrainsToBasin)
                drain_drop_off = self.ScenarioWaterLevels[scenario][j] + max(self.ToBasinRelativeElevation, 0) -\
                    basins_dict[drain_to_basin].ScenarioWaterLevels[scenario][j]

            except ValueError:
                print('wrong DrainToBasin input', self.DrainsToBasin)
                return
        return [drain_to_basin, drain_drop_off]


class Contour(object):

    def __init__(self, code, min_height, surface_area, landuse_dict, landuse_value_dict, population):

        self.Code = code
        self.MinHeight = min_height
        self.SurfaceArea = surface_area
        self.LandUseAreaDict = landuse_dict
        self.LandUseValueDict = landuse_value_dict
        self.Population = population

    def get_contour_damage(self, water_level, damagecurves):
        """

        :param water_level:
        :param damagecurves:
        :return:
        """

        inundation_index = int(round(water_level * 10))   # round to closest 0.10, get index
        contour_damage = 0
        for landuse in self.LandUseAreaDict:
            if self.LandUseAreaDict[landuse] != 0:
                if inundation_index >= len(damagecurves.DevPatternFactors[landuse]):
                    portion_damage = 1
                else:
                    portion_damage = damagecurves.DevPatternFactors[landuse][inundation_index]
                landuse_damage = portion_damage * self.LandUseValueDict[landuse]
                contour_damage += landuse_damage

        return contour_damage


class StructuralMeasure(object):
    """
    example of object class
    """
    Version = "Ver 0.0"
    Type = "Structural"

    def __init__(self, name, code, layer, loc, cconstant, cvariable, repfactor, breachfactor, htunit="m +MSL",
                 slope="None", irribaren="None", costunit='Million', add_strength=1.0, active=False):
        """

        :param name: name of the measure
        :param code: Short code, unique for every measure
        :param layer:
        :param loc:
        :param ht:
        :param cconstant:
        :param cvariable:
        :param repfactor:
        :param breachfactor:
        :param slope:
        :param irribaren:
        :param costunit:
        :param active: Boolean to indicate whether the measure should be used for screening.
        """
        self.Name = name
        self.Code = code
        self.Layer = layer
        self.Location = loc
        self.Height = None
        self.HeightUnits = htunit
        self.Slope = slope
        self.Irribaren = irribaren
        self.CostConstant = cconstant
        self.CostVariable = cvariable
        self.CostUnits = costunit
        self.ReplacementFactor = repfactor  # part of construction costs that it takes to replace the structure
        self.BreachFactor = breachfactor  # part of defense length that is assumed lost when the flood defense 'fails'
        self.AddStrength = add_strength
        self.HeightFail = None
        self.ConstructionCost = 0
        self.RepairCost = 0
        self.Active = active

    def plot_fragility_curve(self):
        """

        :return:
        """
        x = [i / 10 for i in range(101)]
        y = ss.norm.cdf(x, loc=self.Height, scale=0.5)
        return plt.plot(x, y)

    def plot_cost_curve(self):
        """

        :return:
        """
        x = [i / 10 for i in range(101)]
        y = [(self.CostConstant + self.CostVariable * (i / 10)) * 1000 for i in range(101)]
        return plt.plot(x, y)

    def get_failure_probability(self, water_level, waveheight):
        """

        :param hydraulic:
        :return:
        """

        return calculate_failure_probability(self, water_level, waveheight)

    def get_overtopping(self, waterlevel, h_barrier, Hs):
        """

        :param waterlevel:
        :param h_barrier:
        :param Hs:
        :return:
        """
        volume_per_meter = calculate_overtopping(self, waterlevel, h_barrier, Hs)
        return volume_per_meter

    def get_cost_construction_repair(self, ground):
        """

        :param ground:
        :return:
        """
        return calculate_cost_construction_repair(self, ground)  # returns 2 values. [construction cost, repair cost]

    def get_height_fail(self):
        return self.Height + self.AddStrength


class StructuralMeasureLand(StructuralMeasure):
    """
    example of object class
    """
    Land_Water = "Land"

    def __init__(self, name, code, layer, loc, cconstant, cvariable, repfactor=1.2, breachfactor=0.2,
                 htunit="m +MSL", costunit='Million', slope=0.25, irribaren=3, core="clay", layermat="sand", layerthick=1, add_strength=1.0, active=False
                 ):
        """

        :param name:
        :param layer:
        :param loc:
        :param ht:
        :param cconstant:
        :param cvariable:
        :param repfactor:
        :param breachfactor:
        :param slope:
        :param irribaren:
        :param core:
        :param layermat:
        :param layerthick:
        :param ht_fail:
        """
        StructuralMeasure.__init__(self, name, code, layer, loc, cconstant, cvariable, repfactor, breachfactor,
                                   htunit, active)
        self.CostUnits = costunit
        self.Slope = slope
        self.CoreMaterial = core
        self.LayerMaterial = layermat
        self.LayerThickness = layerthick
        self.AddStrength = add_strength
        self.HeightFail = None
        self.Irribaren = irribaren
        self.ConstructionCost = 0
        self.RepairCost = 0


class StructuralMeasureWater(StructuralMeasure):
    """
    example of object class
    """
    Land_Water = "Water"

    def __init__(self, name, code, layer, loc, cconstant, cvariable, repfactor=1.2, breachfactor=1.0, htunit = 'm', slope=1,
                 irribaren=5.0, barriermat="steel", add_strength=1.8, active=False):
        """

        :param name:
        :param layer:
        :param loc:
        :param ht:
        :param cconstant:
        :param cvariable:
        :param repfactor:
        :param breachfactor:
        :param slope:
        :param irribaren:
        :param barriermat:
        :param ht_fail:
        """
        StructuralMeasure.__init__(self, name, code, layer, loc, cconstant, cvariable, repfactor, breachfactor,
                                   htunit, active)

        self.Slope = slope
        self.BarrierMaterial = barriermat
        self.HeightFail = None
        self.AddStrength = add_strength
        self.Irribaren = irribaren
        self.ConstructionCost = 0
        self.RepairCost = 0


class LineOfDefense(object):

    """
    example of object class
    """
    Version = "Ver 0.4"

    def __init__(self, name, typ, sequence, width_unit="m", height_unit='m+MSL', depth_unit='m', fdlocations=None, used_measure=None, incoming_basin=None):
        """

        :param name:
        :param width_unit:
        """
        self.Name = name
        self.Type = typ
        self.Sequence = sequence
        self.WidthUnit = width_unit
        self.HeightUnit = height_unit
        self.DepthUnit = depth_unit
        self.FDLocations = fdlocations
        self.UsedMeasure = used_measure
        self.ScenarioProbabilities = None
        self.Scenarios = None
        self.IncomingBasin = incoming_basin

    def get_overtopping(self, water_level, ht, Hs):
        volume_per_meter = calculate_overtopping(self, water_level, ht, Hs)
        return volume_per_meter

    def get_scenarios(self):
        """


        :param strategy         :   dictionary on strategy measures
        :param layer            :   string on which layer is meant. 'south'/'north'

        :return                 :   scenarios
        """

        scenarios = {}

        outer_situation = [{'East': 'hold', 'West': 'hold'}, {'East': 'fail', 'West': 'hold'},
                           {'East': 'hold', 'West': 'fail'}, {'East': 'fail', 'West': 'fail'}]
        inner_situation = [{'Inland road': 'hold'}, {'Inland road': 'fail'}]

        for scenario in outer_situation:
            for k in range(len(inner_situation)):
                code = str(outer_situation.index(scenario)+1) + "." + str(k + 1)
                scenarios[code] = Scenario(code, {"Coast": scenario, "Inland": inner_situation[k]})

        self.Scenarios = scenarios


class LOD_Land(LineOfDefense):

    """
    example of object class
    """

    def __init__(self, name, typ, sequence, basin_codes_widths, width, height, incoming_basin=None, width_unit='m', height_unit="m", slope=0.25, irribaren=1.25, used_measure=None):
        """
        :param name:
        :param width:
        :param slope:
        :param irribaren:
        """

        LineOfDefense.__init__(self, name, typ, sequence)
        self.IncomingBasin = incoming_basin
        self.WidthUnit = width_unit
        self.HeightUnit = height_unit
        self.BasinCodesWidths = basin_codes_widths
        self.Width = width
        self.Height = height
        self.Slope = slope
        self.Irribaren = irribaren
        self.UsedMeasure = used_measure


class LOD_Water(LineOfDefense):
    """
       example of object class
       """

    def __init__(self, name, typ, sequence, basincodes, width, depth, length, incoming_basin=None, width_unit='m', height_unit="m", depth_unit="m", length_unit='m', slope=0.25, irribaren=1.25, chezy=37, used_measure=None):
        """

        :param name:
        :param width:
        :param depth:
        :param length:
        :param depth_unit:
        :param slope:
        :param irribaren:
        :param chezy:
        """

        LineOfDefense.__init__(self, name, typ, sequence)
        self.BasinCodes = basincodes
        self.Width = width
        self.Depth = depth
        self.Length = length  # characteristic length of the inlet from sea-bay or bay-city
        self.IncomingBasin = incoming_basin
        self.WidthUnit = width_unit
        self.HeightUnit = height_unit
        self.DepthUnit = depth_unit
        self.LengthUnit = length_unit
        self.Slope = slope
        self.Irribaren = irribaren
        self.Chezy = chezy
        self.UsedMeasure = used_measure


class ProtectedArea(object):
    """
    example of object class
    """
    Version = "0.0"

    def __init__(self, name, typ, sequence, basincodes, list_surface_area=[], surface_area_zero=0):
        """

        :param name:
        :param list_surface_area:
        :param surface_area_zero:
        """

        self.Name = name
        self.Type = typ
        self.Sequence = sequence
        self.BasinCodes = basincodes
        self.ListSurfaceArea = list_surface_area
        self.SurfaceAreaZero = surface_area_zero
        self.PDLocations = []

    def create_list_surface_area(self, dict_basin):
        """

        :param dict_basin:
        :return:
        """

        self.ListSurfaceArea = []
        for contour in range(1, 31):
            extra_surf = []
            for basin in dict_basin:
                try:
                    extra_surf.append(dict_basin[basin].Contours[contour].SurfaceArea)
                except ValueError:
                    continue

            self.ListSurfaceArea.append(self.SurfaceAreaZero + np.sum(extra_surf))

        for i in range(1, len(self.ListSurfaceArea)):
            if self.ListSurfaceArea[i] == self.SurfaceAreaZero:
                self.ListSurfaceArea[i] = self.ListSurfaceArea[i - 1]

        return self.ListSurfaceArea


class UnprotectedArea(ProtectedArea):

    def __init__(self, name, typ, sequence, basincodes):

        ProtectedArea.__init__(self, name, typ, sequence, basincodes, list_surface_area=[], surface_area_zero=0)


class NatureBased(object):
    """
    example of object class
    """

    Type = 'Nature-based Solution'

    def __init__(self, name, code, loc, effect, factor, cost, active=False):
        """

        :param name:
        :param loc:
        :param effect:
        :param factor:
        :param cost:
        """
        self.Name = name
        self.Code = code
        self.Location = loc
        self.Effect = effect
        self.ImpactFactor = factor
        self.ConstructionCost = cost
        self.Active = active


class EmergencyMeasure(object):
    """
    example of object class
    """
    Type = 'Emergency'

    def __init__(self, name, code, change_basins, effect, cost, active):

        self.Name = name
        self.Code = code
        self.ChangeBasins = change_basins
        self.Effect = effect
        self.ConstructionCost = cost
        self.Active = active


class DrainageMeasure(object):

    Type = 'Drainage'

    def __init__(self, name, code, change_basins, cost, active=False):

        self.Name = name
        self.Code = code
        self.ChangeBasins = change_basins
        self.ConstructionCost = cost
        self.Active = active


class RetentionMeasure(object):

    Type = 'Retention'

    def __init__(self, name, code, change_basins, cost, active=False):

        self.Name = name
        self.Code = code
        self.ChangeBasins = change_basins
        self.ConstructionCost = cost
        self.Active = active


class Scenario(object):

    def __init__(self, code, sit, prob=None):

        self.Code = code
        self.Situation = sit
        self.Probability = prob


class Impact(object):

    def __init__(self):

        self.ExpectedBasinDamages = {}
        self.ExpectedBasinExposedPop = {}
        self.TotalExpectedDamage = 0
        self.TotalExpectedExposedPop = 0

    def run_impact_calculations(self, region_layout, strategy, damage_curves):

        for basin in region_layout.Basins:
            self.ExpectedBasinDamages[basin] = 0
            self.ExpectedBasinExposedPop[basin] = 0
            for scenario in strategy.ListScenarios:
                self.ExpectedBasinDamages[basin] += region_layout.Basins[basin].get_basin_damage(scenario, damage_curves, strategy.ScenarioProbabilities[scenario])  # stores damage per basin per scenario at Basin type. Expected basin damage (over all scenarios) is accumulated here
                self.ExpectedBasinExposedPop[basin] += region_layout.Basins[basin].get_exposed_population(scenario, strategy.ScenarioProbabilities[scenario])
            self.TotalExpectedDamage += self.ExpectedBasinDamages[basin]
            self.TotalExpectedExposedPop += self.ExpectedBasinExposedPop[basin]

    def show_results(self, region_layout, inflow_rain,inflow_storm, outflow_drain, outflow_infiltration, total_volume_in_system, total_in, total_end, return_period_storm, return_period_rain, struc_measure_east, struc_measure_west, struc_measure_inland,
                           drainage_measure_1, drainage_measure_2, retention_measure_1, retention_measure_2, emergency_measure_1, emergency_measure_2, h_struc_measure_east, h_struc_measure_west, h_struc_measure_inland):

        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        collabel_basins = ('Basin', 'Max water level [1.1]', 'max water level [4.2]', 'Expected damage', 'Exposed population')
        results_basins = []
        for basin in region_layout.Basins:
            results_basins.append([region_layout.Basins[basin].Name, round(region_layout.Basins[basin].ScenarioMaxWaterLevels['1.1'], 2), round(region_layout.Basins[basin].ScenarioMaxWaterLevels['4.2'], 2),
                                   round(self.ExpectedBasinDamages[basin]), round(self.ExpectedBasinExposedPop[basin])])

        collabel_overall = ('data', 'value (scenario=1.1)', 'Value (scenario=4.2)')
        results_overall = [['Inflow rain [m^3]', round(inflow_rain['1.1']), round(inflow_rain['4.2'])],
                           ['Inflow storm [m^3]', round(inflow_storm['1.1']), round(inflow_storm['4.2'])],
                           ['Outflow drain [m^3]', round(outflow_drain['1.1']), round(outflow_drain['4.2'])],
                           ['Outflow infiltration [m^3]', round(outflow_infiltration['1.1']),
                            round(outflow_infiltration['4.2'])],
                           ['Volume in system [m^3]', round(total_volume_in_system['1.1']),
                            round(total_volume_in_system['4.2'])],
                           ['Total in [m^3]', round(total_in['1.1']), round(total_in['4.2'])],
                           ['Total end [m^3]', round(total_end['1.1']), round(total_end['4.2'])],
                           ['Volume loss [%]', round(((total_in['1.1'] - total_end['1.1']) / (total_in['1.1'] + 1)) * 100, 4),  # total_in + 1 necessary to avoid dividing by zero
                            round(((total_in['4.2'] - total_end['4.2']) / (total_in['4.2'] + 1)) * 100, 4)],
                           ['Total damage [Million USD] ', round(self.TotalExpectedDamage) / 1000000, ''],
                           ['Total exposed population', round(self.TotalExpectedExposedPop), '']]

        # ax1.axis('tight')
        ax1.axis('off')
        the_left_table = ax1.table(cellText=results_overall, colLabels=collabel_overall, loc='center',
                              colWidths=[0.4, 0.3, 0.3], cellLoc='right')
        the_left_table.auto_set_font_size(False)
        the_left_table.set_fontsize(12)
        the_left_table.scale(1, 4)

        ax2.axis('off')
        the_right_table = ax2.table(cellText=results_basins, colLabels=collabel_basins, loc='center',
                                    colWidths=[0.15, 0.25, 0.25, 0.25, 0.25], cellLoc='right')
        the_right_table.auto_set_font_size(False)
        the_right_table.set_fontsize(10)
        the_right_table.scale(1, 2.3)

        plt.suptitle('Main results Beira simulation\n'
                     'Return period storm: {} year, rain {} year\n'
                     'Structural measures: East: {} ({} m height), West: {} ({} m height), Inland: {} ({} m high)\n'
                     'Drainage measure: {}'.format(return_period_storm, return_period_rain, struc_measure_east,
                                                   h_struc_measure_east, struc_measure_west, h_struc_measure_west,
                                                   struc_measure_inland, h_struc_measure_inland, drainage_measure_1))

        plt.show()


def plot_help(leg_loc, xl, xu, yl, yu):
    plt.legend(loc=leg_loc)
    axes = plt.gca()
    axes.set_xlim(xl, xu)
    axes.set_ylim(yl, yu)
    axes.set_xlabel('Time (hours)')
    axes.set_ylabel('Height (meters)')


class DamageFunctions(object):
    """
    This object uses the Global flood depth-damage functions by Huizinga et al. (2017) to make Damage curves.
    It is called from the simulation_data_<case>  - function: load_damage_curves(continent,country,modeltype, exchange),
    where the information is loaded into the model through dictionaries 'factors' and 'maxvalues'.

    Example:
    damage_curves = load_damage_curves('AFRICA','Mozambique','Object based',1,30)

    The values from Huizinga et al. are in euros for reference year 2010 with an exchange rate of 1 USD = 0.77 euro.
    """
    def __init__(self, factors, maxvalues, exchange):

        self.DevPatternFactors = {}
        self.Factors = factors
        self.MaxValues = maxvalues
        self.Functions = {}
        for landuse in ['Residential', 'Commercial', 'Industrial', 'Transport', 'Infrastructure', 'Agriculture']:
            try:
                function_tmp = [factor*self.MaxValues[landuse]*exchange for factor in self.Factors[landuse]]
                heights = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]
                new_heights = np.arange(0, 6, 0.1)
                self.Functions[landuse] = calculate_interpolate(function_tmp, heights, new_heights)
                self.Factors[landuse] = calculate_interpolate(list(self.Factors[landuse]), heights, new_heights)
            except TypeError:
                pass

        # Align damage curves Beira
        self.DevPatternFactors = {'DEV_1': self.Factors['Agriculture'],
                                  'DEV_2': self.Factors['Residential'],
                                  'DEV_3': self.Factors['Residential'],
                                  'DEV_5': self.Factors['Industrial'],
                                  'DEV_7': self.Factors['Residential']}

# interx = [(basin,flow_interbasin[basin]['1.1'][i]) for basin in flow_interbasin.keys()]

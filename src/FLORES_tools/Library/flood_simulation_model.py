from __future__ import (absolute_import, division,
                    print_function, unicode_literals)

from .simulation_calculations_beira import run_hydraulic_calculation
from .simulation_definitions_beira import Impact
from .simulation_data_beira import (load_basins, load_layers, load_basin_borders, load_drainage_capacities, load_measures, load_damage_curves, load_hydraulic_conditions,
                                    get_hydraulic_boundary_conditions, get_region_layout, get_strategy)
import numpy as np
from scipy.integrate import trapz

class FloodSimModel:

    def __init__(self):
        """"
        Main FLORES model for fast use in notebooks. Needs input data library and input specific for one simulation

        """

        self.SourcesDict = {}
        self.BasinsMaster = None
        self.LayersMaster = None
        self.AllMeasures = None
        self.HydraulicConditionsMaster = None
        self.DamageCurves = None
        self.BasinBorders = None
        self.DrainageMaster = None
        self.AllDataSources = ['basins', 'layers', 'basin_borders', 'basin_drainage', 'population', 'structures',
                               'damage_curves', 'measures',   'hazard_rain', 'hazard_surge', 'climate_scenarios',
                               'urban_development_scenarios']
        self.ActiveMeasures = {}

    def save_source(self, datatype, datasource, check='no'):

        if datatype not in self.AllDataSources:
            print("Wrong choice for datatype, please choose from:  {}".format(str(self.AllDataSources)))
            return

        if datatype in self.SourcesDict:
            print('Datasource was already defined. new data source overwrites the previous one.')
        self.SourcesDict[datatype] = datasource

        if check == 'yes':
            self.check_data_sources()
        return

    def check_data_sources(self):

        if all(k in self.SourcesDict for k in self.AllDataSources):
            print('All sources defined. Importing datasets.')
            self.import_data()
        return

    def import_data(self):

        self.BasinsMaster = load_basins(self.SourcesDict['basins'],
                                        self.SourcesDict['urban_development_scenarios'])
        self.LayersMaster = load_layers(self.SourcesDict['layers'], self.BasinsMaster)
        self.BasinBorders = load_basin_borders(self.BasinsMaster, self.SourcesDict['basin_borders'])
        self.DrainageMaster = load_drainage_capacities(self.BasinsMaster, self.SourcesDict['basin_drainage'],
                                                       small_channel=6, mid_channel=10, large_channel=35, low_drain=0,
                                                       mid_drain=2, high_drain=4)
        self.AllMeasures = load_measures(self.SourcesDict['measures'])
        self.DamageCurves = load_damage_curves(self.SourcesDict['damage_curves'], 'AFRICA',
                                               'Mozambique', 'Object based', 1.30)
        self.HydraulicConditionsMaster = load_hydraulic_conditions(self.SourcesDict['hazard_surge'],
                                                                   self.SourcesDict['hazard_rain'])

        self.activate_measures(self.ActiveMeasures)
        return

    def define_active_measures(self, **kwargs):

        for key, value in kwargs.items():
            self.ActiveMeasures[key] = value
        return

    def activate_measures(self, active_measures_dict):

        if self.AllMeasures is None:
            print("List of measures not loaded yet. Use '.import_data('measures', /path/to/file)'")
            return
        for measure, value in active_measures_dict.items():
            self.AllMeasures[measure].Active = value

        for measure in self.AllMeasures:
            if self.AllMeasures[measure].Active is False:
                del self.AllMeasures[measure]
        return

    def run_simulation(self, input_data):

        return_period_storm = input_data.ReturnPeriodStormSurge
        return_period_rain = input_data.ReturnPeriodRainfall
        input_scenario_climate = input_data.Scenario
        input_scenario_development = 'none'
        chosen_measures = input_data.ChosenStrategy



        # start of model, loads simulation-specific data
        hydraulic = get_hydraulic_boundary_conditions(self.HydraulicConditionsMaster, return_period_storm,
                                                           return_period_rain,
                                                           input_scenario_climate)
        region_layout = get_region_layout(self.BasinsMaster, self.LayersMaster, input_scenario_development)
        strategy = get_strategy(master=self.AllMeasures, region_layout=region_layout,
                                chosen_measures=chosen_measures, structural_measures_height=input_data.StructuralHeights)
        impact = Impact()

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
        impact.run_impact_calculations(region_layout, strategy, self.DamageCurves)
        risk_reduction = impact.TotalExpectedDamage
        construction_cost = total_cost
        # pdb.set_trace()
        affected_pop_reduction = impact.TotalExpectedExposedPop

        return risk_reduction, construction_cost, affected_pop_reduction

    def screening_simulation_model(self, return_period_storm_surge, return_period_rainfall, climate_scenario, urban_development_scenario, **kwargs):

        # added so we can add scenario information into the replicator
        if return_period_rainfall == 'INFO':
            scenario_info = str(climate_scenario) + ',' + str(urban_development_scenario)
            return [scenario_info] * 3
        #print('surge: {}, rain: {}'.format(return_period_storm_surge,return_period_rainfall))
        strategy = []
        substring_height = 'height'
        substring_scenario = 'scenario'
        structural_heights = {}
        for lever, value in kwargs.items():
            if lever in self.AllMeasures:  # lever is a measure
                if value is True:
                    strategy.append(lever)
            elif substring_height in lever:  # lever is a measure height with boundaries
                measure_code = lever.split('-')[1]
                structural_heights[measure_code] = float(value)
            elif substring_scenario in lever:
                pass
            else:
                print('wrong measure chosen: {}'.format(lever))

        sim_imput = SimulationInput(return_period_storm_surge, return_period_rainfall, strategy, climate_scenario,
                                    structural_heights)
        risk_reduction, construction_cost, affected_pop_reduction = self.run_simulation(sim_imput)
        return risk_reduction, construction_cost, affected_pop_reduction



class SimulationInput:

    def __init__(self, return_period_storm_surge, return_period_rainfall, flood_risk_reduction_strategy, future_scenario, structural_heights):

        self.ReturnPeriodStormSurge = int(return_period_storm_surge)
        self.ReturnPeriodRainfall = int(return_period_rainfall)
        self.ChosenStrategy = flood_risk_reduction_strategy
        self.Scenario = future_scenario
        self.StructuralHeights = {}

        if self.ChosenStrategy is None:
            self.ChosenStrategy = []
        elif type(self.ChosenStrategy) is not list:
            self.ChosenStrategy = list(self.ChosenStrategy)

        for code, height in structural_heights.items():
            self.StructuralHeights[code] = height
        return



def process_risk(data_risk):
    D0_source = {'low,low': 9800000,
                 'low,high': 24092900,  # not calibrated. don't use
                 'high,low': 14980000,
                 'high,high': 31478200  # not calibrated. don't use
                 }

    scenario_info = data_risk.pop()
    D0 = D0_source[scenario_info]  # Base case, storm: 0 year
    data_risk.append(0)  # running the simulation without storm or rain is skipped
    Prob_rain = [0, 0.01, 0.02, 0.1, 0.2, 1]
    Prob_storm = [0, 0.01, 0.02, 0.1, 0.2, 1]
    runs_per_hazard = len(Prob_rain) - 1
    data_risk_array = np.reshape(data_risk, (runs_per_hazard, runs_per_hazard))

    risk_conditional_storm = []
    for count_storm, p_storm in enumerate(Prob_storm[0:5]):
        tmp_conditional_damages = data_risk_array[count_storm]
        conditional_damages = np.append(tmp_conditional_damages, tmp_conditional_damages[-1])
        risk_conditional_storm.append(trapz(conditional_damages, Prob_rain))
    risk_conditional_storm.append(risk_conditional_storm[-1])
    risk = trapz(risk_conditional_storm, Prob_storm)
    risk_reduction = (D0 - risk) / D0

    return risk_reduction


def process_affected_people(data_people):
    P0_source = {'low,low': 32800,
                 'low,high': 80100,  # not calibrated. don't use
                 'high,low': 48700,
                 'high,high': 112100  # not calibrated. don't use
                 }

    scenario_info = data_people.pop()
    P0 = P0_source[scenario_info]
    data_people.append(0)

    Prob_rain = [0, 0.01, 0.02, 0.1, 0.2, 1]
    Prob_storm = [0, 0.01, 0.02, 0.1, 0.2, 1]
    runs_per_hazard = len(Prob_rain) - 1
    data_people_array = np.reshape(data_people, (runs_per_hazard, runs_per_hazard))

    people_conditional_storm = []
    for count_storm, p_storm in enumerate(Prob_storm[0:5]):
        tmp_conditional_people = data_people_array[count_storm]
        conditional_people = np.append(tmp_conditional_people, tmp_conditional_people[-1])
        people_conditional_storm.append(trapz(conditional_people, Prob_rain))

    people_conditional_storm.append(people_conditional_storm[-1])
    affected_population = trapz(people_conditional_storm, Prob_storm)
    affected_population_reduction = (P0 - affected_population) / P0

    return affected_population_reduction


def pick_one(data):
    return data[0]


def pick_50(data):
    return data[2]

def flores_simulation_for_screening( return_period_storm_surge, return_period_rainfall, climate_scenario, simulation_model,  **kwargs):

    strategy = []
    substring_height = 'height'
    structural_heights = {}
    for lever, value in kwargs.items():
        if lever in simulation_model.AllMeasures:  #lever is a measure
            if value is True:
                strategy.append(lever)
        elif substring_height in lever:      #lever is a measure height with boundaries
            measure_code = lever.split('-')[1]
            structural_heights[measure_code] = float(value)
        else:
            print('wrong measure chosen: {}'.format(lever))

    sim_imput = SimulationInput(return_period_storm_surge, return_period_rainfall, strategy, climate_scenario, structural_heights)
    risk_reduction, construction_cost, affected_pop_reduction = simulation_model.run_simulation(sim_imput)
    return risk_reduction, construction_cost, affected_pop_reduction





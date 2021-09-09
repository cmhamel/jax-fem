import os
import yaml
from art import tprint
import argparse
from physics import SteadyStateHeatConduction


if __name__ == '__main__':

    # print header and version
    #
    tprint('tardigrade')

    # parse command line inputs and ensure input file is given
    #
    parser = argparse.ArgumentParser(description='Read in an input file')
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()
    input_file = args.input

    # read input file as yaml dictionary
    #
    with open(input_file, 'r') as stream:
        try:
            input_settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise Exception('Error in input file')

    print(input_settings)

    general = input_settings['general']
    physics = input_settings['physics']

    for key in physics.keys():
        if key.lower() == 'heat_transfer':
            time_dependence = physics[key]['time_dependence']
            if time_dependence == 'steady_state':
                heat_transfer = SteadyStateHeatConduction(general['number_of_dimensions'],
                                                          physics[key]['mesh'])
                print(heat_transfer)
            elif time_dependence == 'transient':
                assert False, 'not supported yet'
            else:
                assert False
        elif key.lower() == 'solid_mechanics':
            assert False, 'not supported yet'
        else:
            assert False, 'Physics not supported currently'


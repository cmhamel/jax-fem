import os
import yaml
from art import tprint
import argparse
from physics import PoissonEquation
from physics import SteadyStateHeatConduction
from physics import RadiativeTransfer
from physics import TransientHeatConduction
from physics import SpeciesTransport
from physics import ExplicitSpeciesTransport
from physics import CahnHilliard
from applications import PhotoChemistry
from jax.config import config
config.update("jax_enable_x64", True)


if __name__ == '__main__':

    # print header and version
    #
    tprint('tardigrade')
    print('Version 0.0')
    print('Authors: Craig Hamel and Lizzy Storm')
    print('\n\n\n')

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

    n_dimensions = input_settings['number_of_dimensions']
    application = input_settings['application']
    physics = input_settings['physics']

    if application == 'photo_chemistry':
        PhotoChemistry(n_dimensions, physics)

    import sys
    sys.exit()

    for key in physics.keys():
        if key.lower() == 'poisson_equation':
            tprint('poisson equation')
            poisson_equation = PoissonEquation(n_dimensions,
                                               physics[key])
        elif key.lower() == 'heat_transfer':
            tprint('heat transfer')
            time_dependence = physics[key]['time_dependence']
            if time_dependence == 'steady_state':
                tprint('steady state')
                heat_transfer = SteadyStateHeatConduction(n_dimensions,
                                                          physics[key])
            elif time_dependence == 'transient':
                tprint('transient')
                heat_transfer = TransientHeatConduction(n_dimensions,
                                                        physics[key])
            else:
                try:
                    assert False
                except AssertionError:
                    raise Exception('Unsupported time dependence')
        elif key.lower() == 'radiative_transfer':
            tprint('radiative transfer')
            radiative_transfer = RadiativeTransfer(n_dimensions,
                                                   physics[key])
        elif key.lower() == 'species_transport':
            tprint('species transport')
            # species_transport = SpeciesTransport(n_dimensions,
            #                                      physics[key])
            species_transport = ExplicitSpeciesTransport(n_dimensions,
                                                         physics[key])
        elif key.lower() == 'cahn_hilliard':
            tprint('cahn-hilliard')
            cahn_hilliard = CahnHilliard(n_dimensions,
                                         physics[key])
        elif key.lower() == 'solid_mechanics':
            assert False, 'not supported yet'
        else:
            try:
                assert False
            except AssertionError:
                raise Exception('Physics not supported currently')


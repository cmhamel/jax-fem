from art import tprint
from parser import Parser
from analysis import Analysis
from jax.config import config
config.update("jax_enable_x64", True)


if __name__ == '__main__':

    # print header and version
    #
    tprint('tardigrade')
    print('Version 0.0')
    print('Authors: Craig Hamel and Lizzy Storm')
    print('\n\n\n')

    parser = Parser()
    input_settings = parser.parse_yaml_file_to_dict()
    analysis = Analysis(input_settings)
    print(analysis)
    analysis.run_analysis()

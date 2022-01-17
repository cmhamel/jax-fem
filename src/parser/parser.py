import argparse
import yaml


class Parser:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Read in an input file')
        parser.add_argument('-i', '--input', required=True)
        # parser.add_argumet()
        args = parser.parse_args()

        self.input_file = args.input

    def parse_yaml_file_to_dict(self):
        with open(self.input_file, 'r') as stream:
            try:
                input_settings = yaml.safe_load(stream)
            except yaml.YAMLError:
                raise Exception('Error in reading yaml input file')

        return input_settings

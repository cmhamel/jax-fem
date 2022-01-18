import numpy as np
import exodus3 as exodus
from util import SuppressStdOutput
from .post_processor_base_class import PostProcessorBaseClass


class ExodusPostProcessor(PostProcessorBaseClass):
    def __init__(self,
                 post_processing_input_settings: dict) -> None:
        super(ExodusPostProcessor, self).__init__(post_processing_input_settings)
        self.exo = self._setup_exodus_file()
        self._setup_nodal_variables()

    def __str__(self) -> str:
        string = '  ' + __class__.__name__ + ':\n'
        string = string + '    Exodus File Name = %s\n' % self.post_processor_input_settings['exodus_file']
        string = string + '    Nodal Variables  = '
        for nodal_variable in self.post_processor_input_settings['requested_outputs']['nodal_variables']:
            string = string + nodal_variable + ' '
        string = string + '\n'
        return string

    def _setup_exodus_file(self) -> str:
        with SuppressStdOutput(suppress_stdout=True):
            exodus_mesh = exodus.exodus(self.post_processor_input_settings['mesh_file'], array_type='numpy')
            exodus_output = exodus_mesh.copy(self.post_processor_input_settings['exodus_file'])
        return exodus_output

    def _setup_nodal_variables(self) -> None:
        nodal_variables = self.post_processor_input_settings['requested_outputs']['nodal_variables']
        self.exo.set_node_variable_number(len(nodal_variables))
        for n, nodal_variable in enumerate(nodal_variables):
            self.exo.put_node_variable_name(nodal_variable, n + 1)

    def write_time_step(self, time_step: int, time_value: float) -> None:
        self.exo.put_time(time_step + 1, time_value)

    def write_nodal_values(self, variable_name: str, time_step: int, variable_values: np.ndarray) -> None:
        self.exo.put_node_variable_values(variable_name, time_step + 1, variable_values)



from .constitutive_model import ConstitutiveModel


class FouriersLaw(ConstitutiveModel):
    def __init__(self, constitutive_model_input_block):
        super(FouriersLaw, self).__init__(constitutive_model_input_block)
        self.check_model_parameters()
        self.k = self.model_parameters['conductivity']

    def __str__(self):
        string = 'Model name = %s\n' % self.model_name
        string = string + 'Model parameters:\n'
        string = string + '\tConductivity: k = %s\n' % self.k
        return string

    def check_model_parameters(self):
        assert self.model_name.lower() == 'fouriers_law'
        assert len(self.model_parameters) == 1
        assert 'conductivity' in self.model_parameters.keys()
        assert self.model_parameters['conductivity'] > 0.0

    def calculate_heat_conduction(self, grad_theta):
        return -self.k * grad_theta

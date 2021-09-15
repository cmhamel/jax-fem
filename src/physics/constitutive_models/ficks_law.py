from .constitutive_model import ConstitutiveModel


class FicksLaw(ConstitutiveModel):
    def __init__(self, constitutive_model_input_block):
        super(FicksLaw, self).__init__(constitutive_model_input_block)
        self.check_model_parameters()
        self.D = self.model_parameters['diffusivity']

    def __str__(self):
        string = 'Model name = %s\n' % self.model_name
        string = string + 'Model parameters:\n'
        string = string + '\tDiffusivity: D = %s\n' % self.D
        return string

    def check_model_parameters(self):
        assert self.model_name.lower() == 'ficks_law'
        assert len(self.model_parameters) == 1
        assert 'diffusivity' in self.model_parameters.keys()
        assert self.model_parameters['diffusivity'] > 0.0

    def calculate_species_flux(self, grad_c):
        return -self.D * grad_c

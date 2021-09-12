

class ConstitutiveModel:
    def __init__(self, constitutive_model_input_block):
        self.constitutive_model_input_block = constitutive_model_input_block
        self.model_name = self.constitutive_model_input_block['model_name']
        self.model_parameters = self.constitutive_model_input_block['model_parameters']

    def __str__(self):
        string = 'Model name = %s\n' % self.model_name
        string = string + 'Model parameters:\n'
        return self.model_name

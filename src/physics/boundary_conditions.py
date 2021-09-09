from pre_processing import GenesisMesh


class BoundaryCondition:
    def __init__(self):
        pass
    

class DirichletBoundaryCondition(BoundaryCondition):
    def __init__(self):
        super(DirichletBoundaryCondition, self).__init__()
        
        
class NeumannBoundaryCondition(BoundaryCondition):
    def __init__(self):
        super(NeumannBoundaryCondition, self).__init__()
    
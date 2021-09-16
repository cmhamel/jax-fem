import jax
import jax.numpy as jnp
from .application import Application
from time_control import TimeControl
from physics import RadiativeTransfer
from physics import ExplicitSpeciesTransport


class PhotoChemistry(Application):
    """
    Application to do a staggered solve on a photochemistry equations

    Light is solved for implicitly every n_I steps to update the light field
    for potential changes in light absorption

    Species transport will be solved for explicitly
    """
    def __init__(self, n_dimension, physics_input_blocks):
        super(PhotoChemistry, self).__init__(n_dimension, physics_input_blocks)

        # unpack input deck block
        #
        self.time_control_block = self.physics_input_blocks['time_control']
        self.radiative_transfer_block = self.physics_input_blocks['radiative_transfer']
        self.species_transport_block = self.physics_input_blocks['species_transport']

        # make objects which build the app physics and time control
        #
        self.time_control = TimeControl(self.time_control_block)
        self.radiative_transfer = RadiativeTransfer(self.n_dimensions, self.radiative_transfer_block)
        self.species_transport = ExplicitSpeciesTransport(self.n_dimensions, self.species_transport_block)

        # set up "sources" in the species transport equation
        #
        self.species_transport.sources = []

        # run the simulation
        #
        self.solve()

    def __str__(self):
        print('Photochemistry app')
        print('Number of dimensions = %s' % self.n_dimensions)

    def solve(self):

        # write initial time first
        #
        # self.radiative_transfer.solve(self.time_control.time_step_number,
        #                               self.time_control.t)
        self.time_control.increment_time()

        while self.time_control.t < self.time_control.time_end:
            # self.radiative_transfer.solve(self.time_control.time_step_number,
            #                               self.time_control.t)

            # update dirichlet bcs
            #
            for n in range(len(self.species_transport.dirichlet_bcs)):
                temp_values = self.species_transport.dirichlet_bcs[n].update_bc_values(time=self.time_control.t)
                self.species_transport.dirichlet_bcs_values = \
                    jax.ops.index_update(self.species_transport.dirichlet_bcs_values, jax.ops.index[n, :], temp_values)

            # print('Time = %s' % self.time_control.t)
            self.species_transport.c_old = \
                self.species_transport.solver.solve(self.time_control,
                                                    self.species_transport.c_old,
                                                    self.species_transport.dirichlet_bcs_nodes,
                                                    self.species_transport.dirichlet_bcs_values)

            if self.time_control.time_step_number % 1000 == 0:
                self.species_transport.post_process_2d(self.time_control.time_step_number,
                                                       self.time_control.t)
            # increment time
            #
            self.time_control.increment_time()

    def post_process(self, time_step, time):
        self.species_transport.post_processor.exo.put_time(time_step, time)
        self.species_transport.post_processor.\
            write_nodal_scalar_variable('c', time_step, jnp.asarray(self.species_transport.c_old))


import jax
import jax.numpy as jnp
from jax import vmap
import arckit

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.hyperneat import HyperNEAT
from tensorneat.algorithm.hyperneat.substrate import MLPSubstrate
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem import BaseProblem
from tensorneat.common import ACT, AGG, State


def preprocess_tasks(train_set, max_size=30):
    inputs = []
    outputs = []
    test_inputs = []
    test_outputs = []
    
    # helper for padded grids to consistent sizes
    def pad_grid(grid):
        arr = jnp.array(grid, dtype=jnp.float32)
        pad_h = max_size - arr.shape[0]
        pad_w = max_size - arr.shape[1]
        
        # We can pad with -1 or 0 (and shift other numbers) since it's 0-9 usually
        # We'll stick with 0 for background logic and just treat it as pixel data
        return jnp.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=-1.0)
    
    for task in train_set:
        for i in range(len(task.train)):
            inputs.append(pad_grid(task.train[i][0]))
            outputs.append(pad_grid(task.train[i][1]))
        test_inputs.append(pad_grid(task.test[0][0]))
        test_outputs.append(pad_grid(task.test[0][1]))
        
    # Stack everything into JAX arrays
    return jnp.stack(inputs), jnp.stack(outputs), jnp.stack(test_inputs), jnp.stack(test_outputs)


class ARCProblem(BaseProblem):
    jitable = True

    def __init__(self, inputs, targets, max_size=30):
        super().__init__()
        # Flatten inputs and targets for the MLP substrate
        self._inputs = inputs.reshape(inputs.shape[0], -1) 
        self._targets = targets.reshape(targets.shape[0], -1)
        self.max_size = max_size

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state, randkey, act_func, params):
        # act_func evaluates a single model. We vmap over the batched inputs
        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self._inputs
        )

        # Cross Entropy or simple Mean Squared Error ignoring the padded -1s
        mask = self._targets != -1.0
        
        # We can evaluate simple MSE here for pixel-wise loss 
        # Only compute loss on the valid pixels
        loss = jnp.mean(jnp.where(mask, (predict - self._targets) ** 2, 0.0))
        
        # Return negative loss since TensorNEAT maximizes fitness
        return -loss

    @property
    def input_shape(self):
        return (self.max_size * self.max_size,)

    @property
    def output_shape(self):
        return (self.max_size * self.max_size,)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        # Calculate final predictability loss 
        fitness = self.evaluate(state, randkey, act_func, params)
        print(f"Best Genome Fitness (Negative Loss): {fitness}")


if __name__ == "__main__":
    train_set, eval_set = arckit.load_data()
    train_in, train_out, test_in, test_out = preprocess_tasks(train_set, max_size=30)
    
    max_size = 30
    grid_pixels = max_size * max_size
    
    # We will use an MLP Substrate mapping grid -> hidden -> grid
    substrate = MLPSubstrate(
        layers=[grid_pixels, 64, grid_pixels], 
    )
    
    pipeline = Pipeline(
        algorithm=HyperNEAT(
            substrate=substrate,
            neat=NEAT(
                pop_size=500,
                species_size=20,
                survival_threshold=0.1,
                compatibility_threshold=1.0,
                genome=DefaultGenome(
                    num_inputs=4,  # Substrate coordinate dimensions (x1, y1, x2, y2)
                    num_outputs=1, # Substrate weight
                    init_hidden_layers=(),
                    output_transform=ACT.tanh,
                ),
            ),
            activation=ACT.tanh,  # Base network activation
            output_transform=ACT.identity, # Output pixel values directly
            activate_time=1, # Feedforward configuration
        ),
        problem=ARCProblem(train_in, train_out, max_size=max_size),
        generation_limit=500,
        fitness_target=-1e-4, 
        seed=42,
    )

    # initialize state
    state = pipeline.setup()
    
    # run until terminate
    state, best = pipeline.auto_run(state)
    
    # show result
    pipeline.show(state, best)
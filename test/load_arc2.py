
import arckit
import jax.numpy as jnp
import jax

train_set, eval_set = arckit.load_data()

def preprocess_tasks(train_set):
    # Extract inputs and outputs from the train examples
    inputs = []
    outputs = []
    test_inputs = []
    test_outputs = []
    for task in train_set:
        for i in range(len(task.train)):
            inputs.append(jnp.array(task.train[i][0]))
            outputs.append(jnp.array(task.train[i][1]))
        test_inputs.append(jnp.array(task.test[0][0]))
        test_outputs.append(jnp.array(task.test[0][1]))
    return inputs, outputs, test_inputs, test_outputs

train_in, train_out, test_in, test_out = preprocess_tasks(train_set)
print(train_in[0], train_out[0])
print(test_in[0], test_out[0])




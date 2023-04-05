import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from numpy import random


def init_params(scale, layer_sizes, rng=random.RandomState(0)):
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params.append(dict(
            weights=jnp.array(scale * rng.randn(n_in, n_out)),
            biases=jnp.array(scale * rng.randn(n_out))))

    return params


def predict(params, inputs):
    activations = inputs
    *hidden, last = params

    for layer in hidden:
        activations = jnp.tanh(
            jnp.dot(activations, layer['weights']) + layer['biases'])

    logits = jnp.dot(activations, last['weights']) + last['biases']

    return logits - logsumexp(logits, axis=1, keepdims=True)


def loss(params, input, target):
    prediction = predict(params, input)

    return -jnp.mean(jnp.sum(prediction * target, axis=1))


@jax.jit
def update(params, input, target, lr=0.01):
    grads = jax.grad(loss)(params, input, target)

    params = jax.tree_map(
        lambda p, g: p - lr *g,
        params,
        grads)

    return params


def accuracy(params, images, labels):
    target_class = jnp.argmax(labels, axis=1)
    predicted_class = jnp.argmax(predict(params, images), axis=1)

    return jnp.mean(predicted_class == target_class)
"""
This work is based on the QDax framework: https://github.com/adaptive-intelligent-robotics/QDax
@article{lim2022accelerated,
  title={Accelerated Quality-Diversity for Robotics through Massive Parallelism},
  author={Lim, Bryan and Allard, Maxime and Grillotti, Luca and Cully, Antoine},
  journal={arXiv preprint arXiv:2202.01258},
  year={2022}
}

This work is also based on the UQD framework: https://github.com/adaptive-intelligent-robotics/Uncertain_Quality_Diversity
@misc{flageat2023uncertain,
      title={Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains}, 
      author={Manon Flageat and Antoine Cully},
      year={2023},
      eprint={2302.00463},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}

This code has been proposed and adapted by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

### Core components of the MAP-Elites algorithm.
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple
from typing import List

import pickle
import tqdm
import os

import jax
import jax.numpy as jnp
import optax
import functools

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.neuroevolution.buffers.buffer import TransitionBuffer
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
    Params,
)

class MAPElites:
    """Core elements of the MAP-Elites algorithm.

    Note: Although very similar to the GeneticAlgorithm, we decided to keep the
    MAPElites class independant of the GeneticAlgorithm class at the moment to keep
    elements explicit.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a MAP-Elites repertoire and compute
            any useful metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        network,
        network_optimizer: optax.OptState,
        l2_rate,
        reevaluations: int = 8
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._network = network # The NN used (MLP, ...)
        self._network_optimizer = network_optimizer # The optimizer used (Adam, SGD, ...) 
        self._reevaluations = reevaluations
        self._l2_rate = l2_rate

    @partial(jax.jit, static_argnames=("self",)) # OK
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """

        # # Score initial genotypes
        # fitnesses, _, descriptors, _, extra_scores, random_key = self._scoring_function(
        #     init_genotypes, random_key
        # )

        fitnesses, fitnesses_var, descriptors, descriptors_var, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # Initialize the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # Get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # print("init genotypes: ", init_genotypes)
        # print("init fitnesses: ", fitnesses)

        # Update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # Initialize random population
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=init_genotypes.shape[0])
        fake_batch = jnp.zeros(shape=(init_genotypes.shape[0], init_genotypes.shape[1]))

        network_parameters = jax.vmap(self._network.init)(keys, fake_batch)
        optimizer = self._network_optimizer.init(network_parameters)

        model_state = {
            'network_params': network_parameters,
            'optimizer': optimizer,
        }

        return repertoire, emitter_state, random_key, model_state

    # Compute scores based only on the scoring function (not the model) for the first generations, so at the beginning
    @partial(jax.jit, static_argnames=("self",)) # OK
    def update_buffer_beginning(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """

        # Generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # print("buffer beginning genotypes: ", genotypes)

        fitnesses, fitnesses_var, descriptors, descriptors_var, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # print("buffer beginning fitnesses: ", fitnesses)

        # Add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # Update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # Update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, genotypes, fitnesses, fitnesses_var, descriptors, descriptors_var, emitter_state, metrics, random_key
    
    # Produce one batch of scores based on current batch of genotypes thank to the scoring function
    @partial(jax.jit, static_argnames=("self",)) # OK
    def generate_batch(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
       
        # Generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        fitnesses, fitnesses_var, descriptors, descriptors_var, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        return genotypes, fitnesses, fitnesses_var, descriptors, descriptors_var, random_key
    
    # Define the loss function
    # @partial(jax.jit, static_argnames=("self", "l2_rate", "reevaluations")) # Doesn't work
    def loss(
        self,
        params, 
        model, 
        geno, 
        preds_func, 
        l2_rate: float = 0.000001, 
        # reevaluations: int = 8,
    ):
        
        # Access parameters of the model and represent them as a dictionary
        parameters = dict(params.items())

        # Predictions using current batcj
        preds_model = jax.vmap(model.apply)(params, geno)

        # multi_preds = []
        # for reeval in range(reevaluations):
        #     # Predictions of the model
        #     preds_model = jax.vmap(model.apply)(params, geno)
        #     multi_preds.append(preds_model)
        # mean_preds_model = jnp.mean(jnp.array(multi_preds), axis=0)
        
        # MSE Loss
        mse_loss = jnp.mean(jnp.square(preds_model - preds_func))

        # L2 Loss (Ridge), Optimal method
        l2_loss = jnp.sum(jnp.array([jnp.sum(jnp.square(w)) for w in jax.tree_util.tree_leaves(parameters["params"])]))

        # Calculating total loss: MSE loss + L2 regularization
        regularized_loss = mse_loss + l2_rate * l2_loss
        
        return regularized_loss, [preds_model, mse_loss]
    
    # Train the model 
    # @partial(jax.jit, static_argnames=("self", "num_epochs", "batch_size", "reevaluations", "l2_rate")) # OK
    def train_model_v2(
        self,
        model_dict,
        training_set: Tuple[Genotype, Fitness, Descriptor],
        num_epochs: int = 300,
        batch_size: int = 512,
        # reevaluations: int = 8,
        l2_rate: float = 0.000001,
    ):
        
        # Training set
        genotypes_train = training_set[0]
        fitness_train = training_set[1].reshape((-1, 1))
        descriptors_train = training_set[2]

        # Number of elements in training and validation sets
        num_train_samples = genotypes_train.shape[0]

        # Number of complete loops with training set
        num_batches_train = num_train_samples // batch_size

        # print("batch_size training: ", batch_size)

        # Keeping track of the training loss
        avg_train_losses = []
        avg_mse_losses = []

        genotypes_batches = jnp.array(jnp.array_split(genotypes_train[:num_batches_train * batch_size], num_batches_train))
        fitnesses_batches = jnp.array(jnp.array_split(fitness_train[:num_batches_train * batch_size], num_batches_train))
        descriptors_batches = jnp.array(jnp.array_split(descriptors_train[:num_batches_train * batch_size], num_batches_train))

        for epoch in range(num_epochs):
                    
            train_losses = jnp.array([None]*num_batches_train)
            mse_losses = jnp.array([None]*num_batches_train)
            
            batch_num = 0

            (model_dict, train_losses, mse_losses, genotypes_batches, fitnesses_batches, descriptors_batches, batch_num, l2_rate), _ = jax.lax.scan(
                self.scan_looping_batches, # Function to scan
                (model_dict, train_losses, mse_losses, genotypes_batches, fitnesses_batches, descriptors_batches, batch_num, l2_rate), # Input args of the function
                (), # Array type returned after the function: tuple
                length=num_batches_train, # Number of scans = for loop iterations
            )

            mse_losses = jnp.array(mse_losses, dtype=jnp.float32)

            avg_train_loss = jnp.mean(jnp.array(train_losses))
            avg_mse_loss = jnp.mean(mse_losses)

            avg_train_losses.append(avg_train_loss)
            array_avg_train_losses = jnp.array(avg_train_losses)
            avg_mse_losses.append(avg_mse_loss)
            array_avg_mse_losses = jnp.array(avg_mse_losses)

        return array_avg_train_losses, array_avg_mse_losses, model_dict
    
    # @partial(jax.jit, static_argnames=("self", "reevaluations", "l2_rate"))
    def looping_batches(
        self,
        model_dict,
        train_losses,
        mse_losses,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        batch_num: int,
        # reevaluations: int = 8,
        l2_rate: float = 0.000001,
    ): # -> Tuple[Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey], Metrics]:
        
        batch_genotypes = genotypes[batch_num]
        batch_fitness = fitnesses[batch_num]
        batch_descriptors = descriptors[batch_num]

        # print("batch_genotypes: ", batch_genotypes.shape[0])

        function_predictions = jnp.concatenate((batch_fitness, batch_descriptors), axis=1)

        # Testing
        loss_val, grads = jax.value_and_grad(self.loss, has_aux=True)(
            model_dict['network_params'], self._network, batch_genotypes, function_predictions, l2_rate
        )

        # Update optimizer state
        updates, model_dict['optimizer'] = self._network_optimizer.update(grads, model_dict['optimizer'], model_dict['network_params'])

        # Update network parameters
        model_dict['network_params'] = optax.apply_updates(model_dict['network_params'], updates)

        train_losses = train_losses.at[batch_num].set(loss_val[0])
        mse_losses = mse_losses.at[batch_num].set(loss_val[1][1])

        batch_num += 1
        
        return train_losses, mse_losses, batch_num
    
    @partial(jax.jit, static_argnames=("self",))
    def scan_looping_batches(
        self,
        carry: Tuple, #Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey],
        unused: Any,
    ):         

        model_dict, train_losses, mse_losses, genotypes_train, fitness_train, descriptors_train, batch_num, l2_rate = carry

        (train_losses, mse_losses, batch_num) = self.looping_batches(
            model_dict=model_dict,
            train_losses=train_losses,
            mse_losses=mse_losses,
            genotypes=genotypes_train,
            fitnesses=fitness_train,
            descriptors=descriptors_train,
            batch_num=batch_num,
            # reevaluations=self._reevaluations,
            l2_rate=self._l2_rate,
        )

        return (model_dict, train_losses, mse_losses, genotypes_train, fitness_train, descriptors_train, batch_num, l2_rate), None

    # Compute scores with model and scoring function and calculate validation loss
    @partial(jax.jit, static_argnames=("self", "batch_size")) # OK
    def check_model(
            self,
            model_dict,
            validation_set: Tuple[Genotype, Fitness, Descriptor],
            batch_size: int = 512,
            # reevaluations: int = 8,
    ): #-> Genotype:
                
        # Validation set
        genotypes_val = validation_set[0]
        fitness_val = validation_set[1]
        descriptors_val = validation_set[2]

        # Number of elements in validation set
        num_val_samples = genotypes_val.shape[0]

        # Number of complete loops with validation set 
        num_batches_val = num_val_samples // batch_size

        # Keeping track of the model predictions
        preds_model = []

        # Looping over batches with batch_size individuals
        for i in range(num_batches_val):
            
            # Creating indices to browse individuals
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Get the current batch of genotypes
            batch_genotypes = genotypes_val[start_idx:end_idx]

            # Predictions using current batch
            predictions_model = jax.vmap(self._network.apply)(model_dict['network_params'], batch_genotypes)

            # Keeping track of the predictions
            preds_model.append(predictions_model)

            # multi_preds = []
            # # Predict using the current batch
            # for reeval in range(reevaluations):
            #     # Predictions of the model
            #     predictions_model = jax.vmap(self._network.apply)(model_dict['network_params'], batch_genotypes)
            #     # print(f"Validation... Number of the batch: {i}/{num_batches_val} | Number of reeval: {reeval}/{reevaluations}")
            #     multi_preds.append(predictions_model)
            # # print("multi_preds: ", multi_preds)
            # mean_preds_model = jnp.mean(jnp.array(multi_preds), axis=0)
            # # batch_preds = jax.vmap(self._network.apply)(self._network_params, batch_genotypes)
            # preds_model.append(mean_preds_model)

        # Concatenate the predictions of all batches from the model
        preds_model = jnp.concatenate(preds_model, axis=0)

        # Extract fitness and descriptors of all predictions from the model
        fitness_val_model = preds_model[:, 0]
        descriptors_val_model = preds_model[:, 1:]

        # Extract fitness and descriptors of all predictions from the scoring function
        fitness_val = fitness_val[0:end_idx].reshape((-1,1))
        descriptors_val = descriptors_val[0:end_idx]

        # Concatenate the predictions of all batches from the scoring function
        preds_func = jnp.concatenate((fitness_val, descriptors_val), axis=1)

        # Calculate validation loss after training all epochs
        avg_val_loss = jnp.mean(jnp.square(preds_model - preds_func))

        return fitness_val_model, fitness_val, descriptors_val_model, descriptors_val, avg_val_loss

    # Compute scores from current batch of genotypes (only one) with model to update repertoire
    @partial(jax.jit, static_argnames=("self")) # OK
    def model_batch_predict(
            self,
            model_dict,
            genotypes: Genotype,
            # reevaluations: int = 8,
    ): # -> Genotype:
        
        preds_model = []
        
        # Predict using the current batch
        predictions_model = jax.vmap(self._network.apply)(model_dict['network_params'], genotypes)

        # Keeping track of the predictions
        preds_model.append(predictions_model)

        # multi_preds = []
        # # Predict using the current batch
        # for reeval in range(reevaluations):
        #     # Predictions of the model using current batch
        #     predictions_model = jax.vmap(self._network.apply)(model_dict['network_params'], genotypes)
        #     # print(f"Current batch outside training... | Number of reeval: {reeval}/{reevaluations}")
        #     multi_preds.append(predictions_model)
        # mean_preds_model = jnp.mean(jnp.array(multi_preds), axis=0)
        # # Predict using the current batch
        # preds_model.append(mean_preds_model)

        # Concatenate the predictions of from the model
        preds_model = jnp.concatenate(preds_model, axis=0)

        # Extract fitness and descriptors of all predictions from the model
        fitness_model = preds_model[:, 0]
        descriptors_model = preds_model[:, 1:]

        return fitness_model, descriptors_model
    
    # Update emitter, repertoire and metrics based on the model predictions 
    @partial(jax.jit, static_argnames=("self",)) # OK
    def update_via_model(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        genotypes: Genotype,
        fitnesses:Fitness,
        descriptors:Descriptor,
        extra_scores
    ):
        
        # print("emitter: ", genotypes.shape)

        # Add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # Update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # Update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics
    
    # ###############################
    # #           UNUSED            #
    # ###############################



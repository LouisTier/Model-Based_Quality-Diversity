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
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._network = network # The NN used (MLP, ...)
        self._network_optimizer = network_optimizer # The optimizer used (Adam, SGD, ...) 
        self._reevaluations = 8
        self._l2_rate = 0.000001

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

        fitnesses, fitnesses_var, descriptors, descriptors_var, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

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
        reevaluations: int = 8,
    ):
        
        # Access parameters of the model and represent them as a dictionary
        parameters = dict(params.items())

        multi_preds = []

        for reeval in range(reevaluations):
            # Predictions of the model
            preds_model = jax.vmap(model.apply)(params, geno)
            multi_preds.append(preds_model)

        # # Copy the genotypes reevaluations times
        # extended_batch_genotypes = jax.tree_util.tree_map(lambda x: jnp.repeat(x, reevaluations), geno)
        # print("loss_function | extended_batch_genotypes: \n", extended_batch_genotypes)
        # # Predictions of the model
        # multi_preds = jax.vmap(model.apply)(params, extended_batch_genotypes)

        mean_preds_model = jnp.mean(jnp.array(multi_preds), axis=0)
        
        # MSE Loss
        mse_loss = jnp.mean(jnp.square(mean_preds_model - preds_func))

        # L2 Loss (Ridge), Optimal method
        l2_loss = jnp.sum(jnp.array([jnp.sum(jnp.square(w)) for w in jax.tree_util.tree_leaves(parameters["params"])]))
        
        # # My method (Debugging)
                    # l2_loss = 0.0
                    # for param in parameters['params'].values(): # Old version
                    #     for value in param.values():
                    #         l2_loss += jnp.sum(jnp.square(value))
        # # AIRL method (Debugging)
        # l2_loss = 0.0
        # for w in jax.tree_util.tree_leaves(parameters["params"]):
        #     l2_loss += jnp.sum(jnp.square(w))

        # Calculating total loss: MSE loss + L2 regularization
        regularized_loss = mse_loss + l2_rate * l2_loss
        
        return regularized_loss, [mean_preds_model, mse_loss]
    
    # Train the model 
    # @partial(jax.jit, static_argnames=("self", "num_epochs", "batch_size", "reevaluations", "l2_rate")) # OK
    def train_model_v2(
        self,
        model_dict,
        training_set: Tuple[Genotype, Fitness, Descriptor],
        num_epochs: int = 300,
        batch_size: int = 512,
        reevaluations: int = 8,
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

        # Keeping track of the training loss
        avg_train_losses = []
        avg_mse_losses = []

        genotypes_batches = jnp.array(jnp.array_split(genotypes_train, num_batches_train))
        fitnesses_batches = jnp.array(jnp.array_split(fitness_train, num_batches_train))
        descriptors_batches = jnp.array(jnp.array_split(descriptors_train, num_batches_train))

        for epoch in range(num_epochs):
                    
            train_losses = jnp.array([None]*num_batches_train)
            mse_losses = jnp.array([None]*num_batches_train)
            
            batch_num = 0

            (model_dict, train_losses, mse_losses, genotypes_batches, fitnesses_batches, descriptors_batches, batch_num, reevaluations, l2_rate), _ = jax.lax.scan(
                self.scan_looping_batches, # Function to scan
                (model_dict, train_losses, mse_losses, genotypes_batches, fitnesses_batches, descriptors_batches, batch_num, reevaluations, l2_rate), # Input args of the function
                (), # Array type returned after the function: tuple
                length=num_batches_train, # Number of scans = for loop iterations
            )

            # # Method before scanning the function
            # for batch_num in range(num_batches_train):
                
            #     train_losses, mse_losses = self.looping_batches(
            #         model_dict=model_dict,
            #         train_losses=train_losses,
            #         mse_losses=mse_losses,
            #         genotypes=genotypes_batches,
            #         fitnesses=fitnesses_batches,
            #         descriptors=descriptors_batches,
            #         # batch_size=batch_size,
            #         batch_num=batch_num,
            #         reevaluations=reevaluations,
            #         l2_rate=l2_rate,
            #         # loss_fn=loss_fn
            #     )

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
        # batch_size: int = 512,
        reevaluations: int = 8,
        l2_rate: float = 0.000001,
        # loss_fn = loss
    ): # -> Tuple[Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey], Metrics]:
        
        # print("Entered looping batches")
                
        # start_idx = batch_num * batch_size
        # end_idx = (batch_num + 1) * batch_size

        # indices = jnp.arange(start_idx, end_idx)
        # batch_genotypes = genotypes[indices]
        # batch_fitness = fitnesses[indices]
        # batch_descriptors = descriptors[indices]

        # batch_genotypes = genotypes[start_idx:end_idx]
        # batch_fitness = fitnesses[start_idx:end_idx]
        # batch_descriptors = descriptors[start_idx:end_idx]

        # batch_genotypes = jax.lax.dynamic_slice(genotypes, (start_idx, 0), (batch_size, genotypes.shape[1]))
        # batch_fitness = jax.lax.dynamic_slice(fitnesses, (start_idx, 0), (batch_size, fitnesses.shape[1]))
        # batch_descriptors = jax.lax.dynamic_slice(descriptors, (start_idx, 0), (batch_size, descriptors.shape[1]))

        batch_genotypes = genotypes[batch_num]
        batch_fitness = fitnesses[batch_num]
        batch_descriptors = descriptors[batch_num]

        function_predictions = jnp.concatenate((batch_fitness, batch_descriptors), axis=1)

        # # Working
        # loss_fn_partial = functools.partial(self.loss, model_dict['network_params'])
        # loss_val, grads = jax.value_and_grad(loss_fn_partial, has_aux=True)(
        #     self._network, batch_genotypes, function_predictions, l2_rate, reevaluations
        # )

        # print("Here")

        # Testing
        loss_val, grads = jax.value_and_grad(self.loss, has_aux=True)(
            model_dict['network_params'], self._network, batch_genotypes, function_predictions, l2_rate, reevaluations
        )

        # print("Here 2")

        # Update optimizer state
        updates, model_dict['optimizer'] = self._network_optimizer.update(grads, model_dict['optimizer'], model_dict['network_params'])

        # Update network parameters
        model_dict['network_params'] = optax.apply_updates(model_dict['network_params'], updates)

        # train_losses.append(loss_val[0])
        # mse_losses.append(loss_val[1][1])
        

        # train_losses[batch_num].append(loss_val[0])
        # mse_losses[batch_num].append(loss_val[1][1])

        train_losses = train_losses.at[batch_num].set(loss_val[0])
        mse_losses = mse_losses.at[batch_num].set(loss_val[1][1])
        # train_losses[batch_num] = loss_val[0]
        # mse_losses[batch_num] = loss_val[1][1]

        batch_num += 1
        
        return train_losses, mse_losses, batch_num
        # return train_losses, mse_losses
    
    @partial(jax.jit, static_argnames=("self",))
    def scan_looping_batches(
        self,
        carry: Tuple, #Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey],
        unused: Any,
    ):         

        # print("Entered scan looping batches")

        model_dict, train_losses, mse_losses, genotypes_train, fitness_train, descriptors_train, batch_num, reevaluations, l2_rate = carry

        # print("reevaluations: ", reevaluations)

        (train_losses, mse_losses, batch_num) = self.looping_batches(
            model_dict=model_dict,
            train_losses=train_losses,
            mse_losses=mse_losses,
            genotypes=genotypes_train,
            fitnesses=fitness_train,
            descriptors=descriptors_train,
            # batch_size=batch_size,
            batch_num=batch_num,
            reevaluations=self._reevaluations,
            l2_rate=self._l2_rate,
        )

        return (model_dict, train_losses, mse_losses, genotypes_train, fitness_train, descriptors_train, batch_num, reevaluations, l2_rate), None

    # @partial(jax.jit, static_argnames=("self", "l2_rate", "reevaluations"))
    # def loss_predictions(
    #     self,
    #     model_dict,
    #     genotypes: Genotype,
    #     fitnesses: Fitness,
    #     descriptors: Descriptor,
    #     batch_num: int,
    #     l2_rate: float,
    #     reevaluations: int,
    # ):
        
    #     batch_genotypes = genotypes[batch_num]
    #     batch_fitness = fitnesses[batch_num]
    #     batch_descriptors = descriptors[batch_num]

    #     function_predictions = jnp.concatenate((batch_fitness, batch_descriptors), axis=1)
        
    #     loss_val, grads = jax.value_and_grad(self.loss, has_aux=True)(
    #         model_dict['network_params'], self._network, batch_genotypes, function_predictions, l2_rate, reevaluations
    #     )

    #     return loss_val, grads
    
    # # @partial(jax.jit, static_argnames=("self")) # -> No
    # def optimizer_upt(
    #         self,
    #         model_dict,
    #         grads
    # ): 
    #     # Update optimizer state
    #     updates, model_dict['optimizer'] = self._network_optimizer.update(grads, model_dict['optimizer'], model_dict['network_params'])

    #     # Update network parameters
    #     model_dict['network_params'] = optax.apply_updates(model_dict['network_params'], updates)
        
    #     return model_dict
    
    # @partial(jax.jit, static_argnames=("self"))
    # def looping_batches(
    #     self,
    #     model_dict,
    #     train_losses,
    #     mse_losses,
    #     loss_val,
    #     # updates,
    #     genotypes: Genotype,
    #     fitnesses: Fitness,
    #     descriptors: Descriptor,
    #     batch_num: int,
    #     reevaluations: int = 8,
    #     l2_rate: float = 0.000001,
    # ): # -> Tuple[Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey], Metrics]:
        
    #     # batch_genotypes = genotypes[batch_num]
    #     # batch_fitness = fitnesses[batch_num]
    #     # batch_descriptors = descriptors[batch_num]

    #     # function_predictions = jnp.concatenate((batch_fitness, batch_descriptors), axis=1)

    #     # # # Working
    #     # # loss_fn_partial = functools.partial(self.loss, model_dict['network_params'])
    #     # # loss_val, grads = jax.value_and_grad(loss_fn_partial, has_aux=True)(
    #     # #     self._network, batch_genotypes, function_predictions, l2_rate, reevaluations
    #     # # )

    #     # # Working - Simpler
    #     # loss_val, grads = jax.value_and_grad(self.loss, has_aux=True)(
    #     #     model_dict['network_params'], self._network, batch_genotypes, function_predictions, l2_rate, reevaluations
    #     # )

    #     # # Update optimizer state
    #     # updates, model_dict['optimizer'] = self._network_optimizer.update(grads, model_dict['optimizer'], model_dict['network_params'])

    #     # # Update network parameters
    #     # model_dict['network_params'] = optax.apply_updates(model_dict['network_params'], updates)

    #     # train_losses = train_losses.at[batch_num].set(loss_val[0])
    #     # mse_losses = mse_losses.at[batch_num].set(loss_val[1][1])

    #     # # Update optimizer state
    #     # updates, model_dict['optimizer'] = self._network_optimizer.update(grads, model_dict['optimizer'], model_dict['network_params'])

    #     # # Update network parameters
    #     # model_dict['network_params'] = optax.apply_updates(model_dict['network_params'], updates)

    #     train_losses = train_losses.at[batch_num].set(loss_val[0])
    #     mse_losses = mse_losses.at[batch_num].set(loss_val[1][1])

    #     batch_num += 1
        
    #     return train_losses, mse_losses, batch_num
    
    # @partial(jax.jit, static_argnames=("self",))
    # def scan_looping_batches(
    #     self,
    #     carry: Tuple, 
    #     unused: Any,
    # ):         

    #     model_dict, train_losses, mse_losses, genotypes_train, fitness_train, descriptors_train, batch_num, reevaluations, l2_rate = carry

    #     loss_val, grads = self.loss_predictions(
    #         model_dict=model_dict,
    #         genotypes=genotypes_train,
    #         fitnesses=fitness_train,
    #         descriptors=descriptors_train,
    #         batch_num=batch_num,
    #         l2_rate=self._l2_rate,
    #         reevaluations=self._reevaluations
    #     )

    #     model_dict = self.optimizer_upt(
    #         model_dict=model_dict,
    #         grads=grads
    #     )

    #     (train_losses, mse_losses, batch_num) = self.looping_batches(
    #         model_dict=model_dict,
    #         train_losses=train_losses,
    #         mse_losses=mse_losses,
    #         loss_val=loss_val,
    #         # updates=updates,
    #         genotypes=genotypes_train,
    #         fitnesses=fitness_train,
    #         descriptors=descriptors_train,
    #         batch_num=batch_num,
    #         reevaluations=self._reevaluations,
    #         l2_rate=self._l2_rate,
    #     )

    #     return (model_dict, train_losses, mse_losses, genotypes_train, fitness_train, descriptors_train, batch_num, reevaluations, l2_rate), None

    # Compute scores with model and scoring function and calculate validation loss
    @partial(jax.jit, static_argnames=("self", "batch_size", "reevaluations")) # OK
    def check_model(
            self,
            model_dict,
            validation_set: Tuple[Genotype, Fitness, Descriptor],
            batch_size: int = 512,
            reevaluations: int = 8,
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

            multi_preds = []

            # Predict using the current batch
            for reeval in range(reevaluations):
                # Predictions of the model
                predictions_model = jax.vmap(self._network.apply)(model_dict['network_params'], batch_genotypes)
                # print(f"Validation... Number of the batch: {i}/{num_batches_val} | Number of reeval: {reeval}/{reevaluations}")
                multi_preds.append(predictions_model)
               
            # # Copy the genotypes reevaluations times
            # print("batch of genotypes: ", batch_genotypes)
            # extended_batch_genotypes = jax.tree_util.tree_map(lambda x: jnp.repeat(x, reevaluations), batch_genotypes)
            # print("extended_batch_genotypes: \n", extended_batch_genotypes)
            # reshaped_batch_genotypes = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (8, 64, 8)), extended_batch_genotypes) # ==> jax.tree_util à modifier
            # print("reshaped_batch_genotypes: \n", reshaped_batch_genotypes)
            # # Predictions of the model
            # # print("Network params: ", model_dict['network_params'])
            # multi_preds = jax.vmap(self._network.apply, in_axes=(None, 0))(model_dict['network_params'], reshaped_batch_genotypes)

            # print("multi_preds: ", multi_preds)
            mean_preds_model = jnp.mean(jnp.array(multi_preds), axis=0)
            
            # batch_preds = jax.vmap(self._network.apply)(self._network_params, batch_genotypes)
            preds_model.append(mean_preds_model)

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
    @partial(jax.jit, static_argnames=("self", "reevaluations")) # OK
    def model_batch_predict(
            self,
            model_dict,
            genotypes: Genotype,
            reevaluations: int = 8,
    ): # -> Genotype:
        
        preds_model = []
        multi_preds = []

        # Predict using the current batch
        for reeval in range(reevaluations):
            # Predictions of the model using current batch
            predictions_model = jax.vmap(self._network.apply)(model_dict['network_params'], genotypes)
            # print(f"Current batch outside training... | Number of reeval: {reeval}/{reevaluations}")
            multi_preds.append(predictions_model)

        # # Copy the genotypes reevaluations times
        # extended_batch_genotypes = jax.tree_util.tree_map(lambda x: jnp.repeat(x, reevaluations), genotypes)
        # print("model_batch_predict | extended_batch_genotypes: \n", extended_batch_genotypes)
        # # Predictions of the model
        # multi_preds = jax.vmap(self._network.apply)(model_dict['network_params'], extended_batch_genotypes)
            
        # print("multi_preds: ", multi_preds)
        mean_preds_model = jnp.mean(jnp.array(multi_preds), axis=0)

        # Predict using the current batch
        # batch_preds = jax.vmap(self._network.apply)(self._network_params, genotypes)
        preds_model.append(mean_preds_model)

        # Concatenate the predictions of from the model
        preds_model = jnp.concatenate(preds_model, axis=0)
        # print("preds_model: ", preds_model)

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

    # @partial(jax.jit, static_argnames=("self",))
    def update(
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

        # Scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

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

        return repertoire, emitter_state, metrics, random_key

    # @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        repertoire, emitter_state, random_key = carry
        (repertoire, emitter_state, metrics, random_key,) = self.update(
            repertoire,
            emitter_state,
            random_key,
        )

        return (repertoire, emitter_state, random_key), metrics
     
    def update_train(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        
        # Emit new genotypes
        genotypes, random_key = self._emitter.emit(repertoire, emitter_state, random_key)
        
        # Score the new genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(genotypes, random_key)
        
        # Add 'transitions' obtained via the scoring function into the container
        self.replay_buffer.add_transitions_2(genotypes, fitnesses, descriptors)

        # Sample a batch of size of the genotype batch transitions from the buffer
        batch_genotypes, batch_fitnesses, batch_BDs, random_key = self.replay_buffer.sample(batch_size=genotypes.shape[0], 
                                                                                            random_key=random_key)

        # Reshaping fitnesses to concatenate them with BDs: mandatory to calculate the loss
        batch_reshaped_fitnesses = batch_fitnesses.reshape((-1, 1))

        # Concatenating fitness and descriptors,
        preds_func = jnp.concatenate((batch_reshaped_fitnesses, batch_BDs), axis=1)
       
        # Defining the loss (MSE) used to train the NN
        def loss(params, genotypes, preds_func):

            # Predictions of the model
            preds_model = jax.vmap(self._network.apply)(params, genotypes)
            # MSE Loss
            mse_loss = jnp.mean(jax.numpy.square(preds_model - preds_func))
            
            return mse_loss, preds_model
        
        # Compute the loss and gradients of the loss with respect to the network parameters
        # Here, loss_val = [mse_loss, preds_model] 
        # 'has_ax' indicated that first returned element is differentiable (mse_loss), the rest is 'auxiliary' data
        loss_val, grads = jax.value_and_grad(loss, has_aux=True)(self._network_params, batch_genotypes, preds_func)
        # Update optimizer state
        updates, self.optimizer = self._network_optimizer.update(grads, self.optimizer, self._network_params)
        # Update network parameters
        self._network_params = optax.apply_updates(self._network_params, updates)

        # Use the neural network to predict the scores for genotypes of current generation after being trained
        preds_model = jax.vmap(self._network.apply)(self._network_params, genotypes)
        fitnesses_pred = preds_model[:, 0]
        descriptors_pred = preds_model[:, 1:]
        
        # Add the new solutions to the repertoire with the predicted scores
        repertoire = repertoire.add(batch_of_genotypes=genotypes, 
                                    batch_of_descriptors=descriptors_pred, 
                                    batch_of_fitnesses=fitnesses_pred, 
                                    batch_of_extra_scores=None
        )

        # Update the emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses_pred,
            descriptors=descriptors_pred,
            extra_scores=None,
        )

        # Calculate and return metrics
        metrics = self._metrics_function(repertoire)
        
        # loss_val[0] corresponds to the MSE loss calculated at each generation
        return repertoire, emitter_state, metrics, random_key, loss_val[0]
    
    def update_train_2(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
        num_batches: int = 50,
        patience: int = 7,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey, float]:
        
        # Emit new genotypes
        genotypes, random_key = self._emitter.emit(repertoire, emitter_state, random_key)
        
        # Score the new genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(genotypes, random_key)
        
        # Add 'transitions' obtained via the scoring function into the container
        self.replay_buffer.add_transitions_2(genotypes, fitnesses, descriptors)

        # Keep track of loss of each batch
        losses = []

        # min_loss = float('inf')
        lowest_loss = self.min_loss
        # print(f"lowest_loss: {lowest_loss}")

        # Train the neural network on multiple batches
        ### Cover whole rep: in range(self.buffer//num of solutions)
        for batch_num in range(num_batches):
            
            # Sample a batch of size of the genotype batch transitions from the buffer
            batch_genotypes, batch_fitnesses, batch_BDs, random_key = self.replay_buffer.sample(batch_size=genotypes.shape[0], 
                                                                                                random_key=random_key)
            # Update min and max values
            if self.min_BD is None:
                self.min_BD = jnp.min(batch_BDs, axis=0)
                self.max_BD = jnp.max(batch_BDs, axis=0)
            else:
                self.min_BD = jnp.minimum(self.min_BD, jnp.min(batch_BDs, axis=0))
                self.max_BD = jnp.maximum(self.max_BD, jnp.max(batch_BDs, axis=0))
            
            # Reshaping fitnesses to concatenate them with BDs: mandatory to calculate the loss
            batch_reshaped_fitnesses = batch_fitnesses.reshape((-1, 1))

            # Concatenating fitness and descriptors,
            preds_func = jnp.concatenate((batch_reshaped_fitnesses, batch_BDs), axis=1)
            
            # Define the loss (MSE) used to train the NN
            def loss(params, genotypes, preds_func):
                # Predictions of the model
                preds_model = jax.vmap(self._network.apply)(params, genotypes)
                # Normalize fitness and BD
                # preds_model_normalized = self.dynamic_normalize(preds_model)
                # MSE Loss
                mse_loss = jnp.mean(jnp.square(preds_model - preds_func))
                return mse_loss, preds_model

            # Compute the loss and gradients of the loss with respect to the network parameters
            loss_val, grads = jax.value_and_grad(loss, has_aux=True)(self._network_params, batch_genotypes, preds_func)
            
            losses.append(loss_val[0])  # Collect the loss for each batch

            # print(f"batch num: {batch_num} || current loss: {loss_val[0]} || minimal loss: {lowest_loss}")

            # Keep track of the minimum loss
            if loss_val[0] < lowest_loss:
                lowest_loss = loss_val[0]
                self.min_loss = lowest_loss
                self.counter = 0 #reset

                # # Update optimizer state
                # updates, self.optimizer = self._network_optimizer.update(grads, self.optimizer, self._network_params)
                # # Update network parameters
                # self._network_params = optax.apply_updates(self._network_params, updates)
            
            else:
                self.counter += 1
                # print(f'Patience: {self.counter} out of {patience}')

            # Check for early stopping
            if self.counter > patience-1:
                # print(f'Patience: {self.counter} out of {patience}')
                # print(f'Stop Training for current generation at batch n° {batch_num}')
                self.counter = 0
                break

            # Update optimizer state
            updates, self.optimizer = self._network_optimizer.update(grads, self.optimizer, self._network_params)
            # Update network parameters
            self._network_params = optax.apply_updates(self._network_params, updates)

        # Use the neural network to predict the scores for genotypes of current generation after being trained
        preds_model = jax.vmap(self._network.apply)(self._network_params, genotypes)
        fitnesses_pred = preds_model[:, 0]
        descriptors_pred = preds_model[:, 1:]
        
        # Add the new solutions to the repertoire with the predicted scores
        repertoire = repertoire.add(batch_of_genotypes=genotypes, 
                                    batch_of_descriptors=descriptors_pred, 
                                    batch_of_fitnesses=fitnesses_pred, 
                                    batch_of_extra_scores=None
        )

        # Update the emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses_pred,
            descriptors=descriptors_pred,
            extra_scores=None,
        )

        # Calculate and return metrics
        metrics = self._metrics_function(repertoire)

        # Calculate the avg loss of all batches for current generation
        losses_array = jnp.array(losses)
        avg_loss = jnp.mean(losses_array)
        
        return repertoire, emitter_state, metrics, random_key, avg_loss, descriptors, descriptors_pred, fitnesses, fitnesses_pred  
    
    def update_train_3(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
        num_batches: int = 50,
        patience: int = 7,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey, float]:
        
        # Emit new genotypes
        genotypes, random_key = self._emitter.emit(repertoire, emitter_state, random_key)
        
        # Score the new genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(genotypes, random_key)
        
        # Add 'transitions' obtained via the scoring function into the container
        # self.replay_buffer.add_transitions_2(genotypes, fitnesses, descriptors)

        # Keep track of loss of each batch
        losses = []

        lowest_loss = self.min_loss

        # Train the neural network on multiple batches
        ### Cover whole rep: in range(self.buffer//num of solutions)
        for batch_num in range(num_batches):
            
            # # Sample a batch of size of the genotype batch transitions from the buffer
            # batch_genotypes, batch_fitnesses, batch_BDs, random_key = self.replay_buffer.sample(batch_size=genotypes.shape[0], 
            #                                                                                     random_key=random_key)
                        
            # # Reshaping fitnesses to concatenate them with BDs: mandatory to calculate the loss
            # batch_reshaped_fitnesses = batch_fitnesses.reshape((-1, 1))

            # # Concatenating fitness and descriptors,
            # preds_func = jnp.concatenate((batch_reshaped_fitnesses, batch_BDs), axis=1)

            batch_genotypes, random_key = repertoire.sample(random_key=random_key, num_samples=genotypes.shape[0])
            # print("batch_genotypes size: ", batch_genotypes.shape)
            # print("batch_genotypes: ", batch_genotypes)

            # Score the new genotypes
            batch_fitnesses, batch_descriptors, batch_extra_scores, random_key = self._scoring_function(batch_genotypes, random_key)
            batch_fitnesses = batch_fitnesses.reshape((-1,1))

            preds_func = jnp.concatenate((batch_fitnesses, batch_descriptors), axis=1)

            # Define the loss (MSE) used to train the NN
            def loss(params, genotypes, preds_func):
                # Predictions of the model
                preds_model = jax.vmap(self._network.apply)(params, genotypes)
                # MSE Loss
                mse_loss = jnp.mean(jnp.square(preds_model - preds_func))
                return mse_loss, preds_model

            # Compute the loss and gradients of the loss with respect to the network parameters
            loss_val, grads = jax.value_and_grad(loss, has_aux=True)(self._network_params, batch_genotypes, preds_func)
            
            losses.append(loss_val[0])  # Collect the loss for each batch

            # Keep track of the minimum loss
            if loss_val[0] < lowest_loss:
                lowest_loss = loss_val[0]
                self.min_loss = lowest_loss
                self.counter = 0 #reset
            
            else:
                self.counter += 1

            # Check for early stopping
            if self.counter > patience-1:
                self.counter = 0
                break

            # Update optimizer state
            updates, self.optimizer = self._network_optimizer.update(grads, self.optimizer, self._network_params)
            # Update network parameters
            self._network_params = optax.apply_updates(self._network_params, updates)

        # Use the neural network to predict the scores for genotypes of current generation after being trained
        preds_model = jax.vmap(self._network.apply)(self._network_params, genotypes)
        fitnesses_pred = preds_model[:, 0]
        descriptors_pred = preds_model[:, 1:]
        
        # # Add the new solutions to the repertoire with the predicted scores
        # repertoire = repertoire.add(batch_of_genotypes=genotypes, 
        #                             batch_of_descriptors=descriptors_pred, 
        #                             batch_of_fitnesses=fitnesses_pred, 
        #                             batch_of_extra_scores=None
        # )

        # # Update the emitter state
        # emitter_state = self._emitter.state_update(
        #     emitter_state=emitter_state,
        #     repertoire=repertoire,
        #     genotypes=genotypes,
        #     fitnesses=fitnesses_pred,
        #     descriptors=descriptors_pred,
        #     extra_scores=None,
        # )

        # Add the new solutions to the repertoire with the predicted scores
        repertoire = repertoire.add(batch_of_genotypes=genotypes, 
                                    batch_of_descriptors=descriptors, 
                                    batch_of_fitnesses=fitnesses, 
                                    batch_of_extra_scores=extra_scores
        )

        # Update the emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # Calculate and return metrics
        metrics = self._metrics_function(repertoire)

        # Calculate the avg loss of all batches for current generation
        losses_array = jnp.array(losses)
        avg_loss = jnp.mean(losses_array)
        
        return repertoire, emitter_state, metrics, random_key, avg_loss, descriptors, descriptors_pred, fitnesses, fitnesses_pred  
    
    def normalize(self, BD):
        min_vals = jnp.min(BD, axis=0)
        max_vals = jnp.max(BD, axis=0)
        normalized_BD = (BD - min_vals) / (max_vals - min_vals)
        return normalized_BD
       
    def dynamic_normalize(self, preds_model):

        normalized_fitness = (preds_model[:, 0] - self.min_fitness) / (self.max_fitness - self.min_fitness)
        normalized_BD = (preds_model[:, 1:] - self.min_BD) / (self.max_BD - self.min_BD)

        normalized_BD = jnp.clip(normalized_BD, self.min_BD, self.max_BD)
        normalized_fitness = jnp.clip(normalized_fitness, self.min_fitness, self.max_fitness)

        normalized_preds_model = jnp.concatenate((normalized_fitness.reshape((-1, 1)), normalized_BD), axis=1)
        return normalized_preds_model

    def train_model(
            self,
            repertoire: MapElitesRepertoire,
            random_key: RNGKey,
            num_epochs: int = 300,
            # num_batches: int = 50,
            patience: int = 7,
            batch_size: int = 512,
    ): 
        
        random_key, subkey = jax.random.split(random_key)

        # Keep track of loss of each batch
        lowest_loss = self.min_loss

        all_fitness = repertoire.fitnesses
        valid_indices = jnp.where(all_fitness != float('-inf'))[0]
        filtered_fitness = all_fitness[valid_indices]
        all_genotypes = repertoire.genotypes
        filtered_genotypes = all_genotypes[valid_indices]
        all_descriptors = repertoire.descriptors
        filtered_descriptors = all_descriptors[valid_indices]

        # Define the train-test split ratio
        train_ratio = 0.7

        # Shuffle the indices
        num_samples = all_genotypes.shape[0]
        shuffled_indices = jax.random.permutation(subkey, num_samples)

        # Calculate the split point
        split_point = int(num_samples * train_ratio)

        # Split the indices into training and validation sets
        train_indices = shuffled_indices[:split_point]
        val_indices = shuffled_indices[split_point:]

        # Split the genotypes, fitness, and descriptors using the indices
        genotypes_train = filtered_genotypes[train_indices]
        genotypes_val = filtered_genotypes[val_indices]
        fitness_train = filtered_fitness[train_indices]
        fitness_val = filtered_fitness[val_indices]
        descriptors_train = filtered_descriptors[train_indices]
        descriptors_val = filtered_descriptors[val_indices]

        avg_train_losses = []
        
        # Calculate the number of batches dynamically
        num_train_samples = genotypes_train.shape[0]
        num_batches_train = num_train_samples // batch_size

        # Lengh of the training
        for epoch in range(num_epochs):

            # print(f"Epoch: {epoch}/{num_epochs}")
            
            train_losses = []

            # Train the neural network on multiple batches
            for batch_num in range(num_batches_train):

                # print(f"Batch: {batch_num}/{num_batches_train}")

                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size

                indices = jnp.arange(start_idx, end_idx)
                # print("indices: ", indices)
                # indices = jax.random.choice(subkey, batch_size, shape=(batch_size,), replace=False)
                batch_genotypes = genotypes_train[indices]
                batch_fitness = fitness_train[indices]
                batch_fitness = batch_fitness.reshape((-1,1))
                batch_descriptors = descriptors_train[indices]

                function_predictions = jnp.concatenate((batch_fitness, batch_descriptors), axis=1)

                # print("function_predictions: ", function_predictions.shape)

                # print("self._network_params: ", self._network_params,)
                
                # Define the loss (MSE) used to train the NN
                def loss(params, geno, preds_func, l2_rate=0.000001):

                    params = dict(params.items())

                    # Predictions of the model
                    preds_model = jax.vmap(self._network.apply)(params, geno)
                    # print("preds_model: ", preds_model.shape)
                    # MSE Loss
                    mse_loss = jnp.mean(jnp.square(preds_model - preds_func))
                    
                    # L2 regularization
                    l2_loss = 0.0
                    for param in self._network_params['params'].values():
                        for value in param.values():
                            l2_loss += jnp.sum(jnp.square(value))

                    # Calculate the total loss by combining MSE loss and L2 regularization
                    regularized_loss = mse_loss + l2_rate * l2_loss

                    return regularized_loss, preds_model

                # random_key, subkey = jax.random.split(random_key)

                # Compute the loss and gradients of the loss with respect to the network parameters
                loss_val, grads = jax.value_and_grad(loss, has_aux=True)(self._network_params, batch_genotypes, function_predictions, l2_rate=0.000001)
                
                # Update optimizer state
                updates, self.optimizer = self._network_optimizer.update(grads, self.optimizer, self._network_params)
                # Update network parameters
                self._network_params = optax.apply_updates(self._network_params, updates)

                train_losses.append(loss_val[0])  # Collect the loss for each batch

            array_train_losses = jnp.array(train_losses)
            avg_train_loss = jnp.mean(array_train_losses)

            avg_train_losses.append(avg_train_loss)
            array_avg_train_losses = jnp.array(avg_train_losses)

            # print(f"epoch: {epoch} | patience: {self.counter} | current loss: {avg_train_loss} | minimal loss: {lowest_loss}")
            
            # # Keep track of the minimum loss
            # if avg_train_loss < lowest_loss:
            #     lowest_loss = avg_train_loss
            #     self.min_loss = lowest_loss
            #     self.counter = 0 #reset
            
            # else:
            #     self.counter += 1
            #     # print(f"counter: {self.counter}/{patience}")

            # # Check for early stopping
            # if self.counter > patience-1:
            #     self.counter = 0
            #     print(f"Stop training after {epoch} epochs out of {num_epochs}")
            #     break

        num_val_samples = genotypes_val.shape[0]
        num_batches_val = num_val_samples // batch_size
        # print("num_batches_val: ", num_batches_val)

        # self._network = self._network(is_training=False)

        preds_model = []

        for i in range(num_batches_val):
            # print(f"iteration: {i}/{num_batches_val}")
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # print(f"start_idx: {start_idx}, end_idx: {end_idx}")

            # Get the current batch of genotypes
            batch_genotypes = genotypes_val[start_idx:end_idx]

            # Predict using the current batch
            batch_preds = jax.vmap(self._network.apply)(self._network_params, batch_genotypes)
            preds_model.append(batch_preds)

        # Concatenate the predictions
        preds_model = jnp.concatenate(preds_model, axis=0)

        # Extract fitness and descriptors
        fitness_val_model = preds_model[:, 0]
        descriptors_val_model = preds_model[:, 1:]

        fitness_val = fitness_val[0:end_idx]
        fitness_val_reshaped = fitness_val.reshape((-1,1))
        descriptors_val = descriptors_val[0:end_idx]

        preds_func = jnp.concatenate((fitness_val_reshaped, descriptors_val), axis=1)

        # Calculate validation loss after training all epochs
        avg_val_loss = jnp.mean(jnp.square(preds_model - preds_func))

        model_state = {
            'network_params': self._network_params,
            'optimizer': self.optimizer, 
        } 

        # print(fitness_val_model.shape, fitness_val.shape, descriptors_val_model.shape, descriptors_val.shape)

        return fitness_val_model, fitness_val, descriptors_val_model, descriptors_val, array_avg_train_losses, avg_val_loss, model_state, random_key

    def split_repertoire(
            self,
            random_key: RNGKey,
            repertoire: MapElitesRepertoire,
            train_ratio: float = 0.7,
    ) -> Tuple[Tuple[Genotype, Fitness, Descriptor], Tuple[Genotype, Fitness, Descriptor], RNGKey]: 
        
        # All cells in the container
        all_fitness = repertoire.fitnesses
        valid_indices = jnp.where(all_fitness != float('-inf'))[0]
        all_genotypes = repertoire.genotypes
        all_descriptors = repertoire.descriptors

        # Only cells non empty
        filtered_genotypes = all_genotypes[valid_indices]
        filtered_fitness = all_fitness[valid_indices]
        filtered_descriptors = all_descriptors[valid_indices]

        # Shuffle indices of non empty cells (randomize individuals)
        random_key, subkey = jax.random.split(random_key)
        num_samples = all_genotypes.shape[0]
        shuffled_indices = jax.random.permutation(subkey, num_samples)

        # Calculate the split point to dissociate train and validation sets
        split_point = int(num_samples * train_ratio)
        train_indices = shuffled_indices[:split_point]
        val_indices = shuffled_indices[split_point:]
        # print("train_indices: ", train_indices)
        # print()
        # print("val_indices: ", val_indices)

        # Training set
        genotypes_train = filtered_genotypes[train_indices]
        fitness_train = filtered_fitness[train_indices]
        descriptors_train = filtered_descriptors[train_indices]

        # Validation set
        genotypes_val = filtered_genotypes[val_indices]
        fitness_val = filtered_fitness[val_indices]
        descriptors_val = filtered_descriptors[val_indices]

        return [genotypes_train, fitness_train, descriptors_train], [genotypes_val, fitness_val, descriptors_val], random_key

    # Compute scores with scoring function and model to compare them based on the repertoire X epochs before training
    # @partial(jax.jit, static_argnames=("self",))
    def predict(
            self,
            repertoire: MapElitesRepertoire,
            batch_size: int = 512,
    ): # -> Genotype:
        
        # All cells in the container
        all_fitness = repertoire.fitnesses
        valid_indices = jnp.where(all_fitness != float('-inf'))[0]
        all_genotypes = repertoire.genotypes
        all_descriptors = repertoire.descriptors

        # Only cells non empty
        filtered_genotypes = all_genotypes[valid_indices]
        filtered_fitness = all_fitness[valid_indices]
        filtered_descriptors = all_descriptors[valid_indices]

        # Number of elements in training and validation sets
        num_train_samples = filtered_genotypes.shape[0]

        # Number of complete loops with training set
        num_batches_train = num_train_samples // batch_size

        # print(f"For this generation, they are {num_batches_train*batch_size} predictions")

        # Keeping track of the model predictions
        preds_model = []

        # Predict scores with the model
        for i in range(num_batches_train):

            # Creating indices to browse individuals
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Get the current batch of genotypes
            batch_genotypes = filtered_genotypes[start_idx:end_idx]

            # Predict using the current batch
            batch_preds = jax.vmap(self._network.apply)(self._network_params, batch_genotypes)
            preds_model.append(batch_preds)

        # Concatenate the predictions of all batches from the model
        preds_model = jnp.concatenate(preds_model, axis=0)

        # Extract fitness and descriptors of all predictions from the scoring function
        filtered_fitness = filtered_fitness[0:end_idx].reshape((-1,1))
        filtered_descriptors = filtered_descriptors[0:end_idx]

        # Concatenate the predictions of all batches from the scoring function
        preds_func = jnp.concatenate((filtered_fitness, filtered_descriptors), axis=1)

        sol_function = [filtered_genotypes[0:end_idx], preds_func]
        sol_model = [filtered_genotypes[0:end_idx], preds_model]

        return sol_function, sol_model
    
    # Train the model 
    # @partial(jax.jit, static_argnames=("self", "num_epochs", "batch_size", "reevaluations")) # Doesn't work, but should be unused
    def train_model_v2_original(
            self,
            training_set: Tuple[Genotype, Fitness, Descriptor],
            num_epochs: int = 300,
            batch_size: int = 512,
            reevaluations: int = 8,
            l2_rate: float = 0.000001,
    ):  #-> Tuple[]:

        # Training set
        genotypes_train = training_set[0]
        fitness_train = training_set[1]
        descriptors_train = training_set[2]

        # Number of elements in training and validation sets
        num_train_samples = genotypes_train.shape[0]

        # Number of complete loops with training set
        num_batches_train = num_train_samples // batch_size

        # Keeping track of the training loss
        avg_train_losses = []
        avg_mse_losses = []

        # Lengh of the training
        for epoch in tqdm.tqdm(range(num_epochs)):
            
            train_losses = []
            mse_losses = []

            # Train the neural network on multiple batches
            for batch_num in range(num_batches_train):
                
                # Creating indices to browse individuals
                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size
                indices = jnp.arange(start_idx, end_idx)

                # Selecting corresponding individuals
                batch_genotypes = genotypes_train[indices]
                batch_fitness = fitness_train[indices].reshape((-1,1))
                batch_descriptors = descriptors_train[indices]

                # Define predictions of scoring function: fitness & BD
                function_predictions = jnp.concatenate((batch_fitness, batch_descriptors), axis=1)

                multi_preds = []

                # Define the loss (MSE) used to train the NN
                # def loss(params, geno, preds_func):
                def loss(params, geno, preds_func, l2_rate=l2_rate):
                    
                    # Access parameters of the model and represent them as a dictionary
                    parameters = dict(params.items())

                    # # Predictions of the model
                    # preds_model = jax.vmap(self._network.apply)(parameters, geno)

                    for reeval in range(reevaluations):
                        # Predictions of the model
                        preds_model = jax.vmap(self._network.apply)(params, geno)
                        # print(f"Training... Number of the batch: {batch_num}/{num_batches_train} | Number of reeval: {reeval}/{reevaluations}")
                        multi_preds.append(preds_model)
                        
                    mean_preds_model = jnp.mean(jnp.array(multi_preds), axis=0)

                    # MSE Loss
                    mse_loss = jnp.mean(jnp.square(mean_preds_model - preds_func))
                    
                    # L2 Loss (Ridge), Optimal method
                    l2_loss = 0.0
                    l2_loss = jnp.sum(jnp.array([jnp.sum(jnp.square(w)) for w in jax.tree_util.tree_leaves(parameters["params"])]))
                    
                    # # My method (Debugging)
                    # l2_loss = 0.0
                    # for param in parameters['params'].values(): # Old version
                    #     for value in param.values():
                    #         l2_loss += jnp.sum(jnp.square(value))

                    # # AIRL method (Debugging)
                    # l2_loss = 0.0
                    # for w in jax.tree_util.tree_leaves(parameters["params"]):
                    #     l2_loss += jnp.sum(jnp.square(w))

                    # Calculating total loss: MSE loss + L2 regularization
                    regularized_loss = mse_loss + l2_rate * l2_loss

                    return regularized_loss, [mean_preds_model, mse_loss]
                    # return l2_loss, preds_model
                    # return mse_loss, preds_model
                
                # Compute the loss and gradients of the loss with respect to the network parameters
                loss_val, grads = jax.value_and_grad(loss, has_aux=True)(self._network_params, batch_genotypes, function_predictions, l2_rate=l2_rate)

                # Update optimizer state
                updates, self.optimizer = self._network_optimizer.update(grads, self.optimizer, self._network_params)

                # Update network parameters
                self._network_params = optax.apply_updates(self._network_params, updates)

                # Collecting the loss of each bach for a given epoch
                train_losses.append(loss_val[0])  
                mse_losses.append(loss_val[1][1])

            mse_losses = jnp.array(mse_losses, dtype=jnp.float32)

            # Calculating the average loss of one epoch (so for all batches)
            avg_train_loss = jnp.mean(jnp.array(train_losses))
            avg_mse_loss = jnp.mean(mse_losses)

            # Collecting the average loss obtained at each epoch
            avg_train_losses.append(avg_train_loss)
            array_avg_train_losses = jnp.array(avg_train_losses)
            avg_mse_losses.append(avg_mse_loss)
            array_avg_mse_losses = jnp.array(avg_mse_losses)

        model_state = {
            'network_params': self._network_params,
            'optimizer': self.optimizer, 
        } 

        return array_avg_train_losses, array_avg_mse_losses, model_state

    # Compute scores with model and scoring function and calculate validation loss
    # @partial(jax.jit, static_argnames=("self", "batch_size", "reevaluations")) # NOT OK, but should be unused
    def check_model_original(
            self,
            validation_set: Tuple[Genotype, Fitness, Descriptor],
            batch_size: int = 512,
            reevaluations: int = 8,
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

            multi_preds = []

            # Predict using the current batch
            for reeval in range(reevaluations):
                # Predictions of the model
                predictions_model = jax.vmap(self._network.apply)(self._network_params, batch_genotypes)
                # print(f"Validation... Number of the batch: {i}/{num_batches_val} | Number of reeval: {reeval}/{reevaluations}")
                multi_preds.append(predictions_model)
                
            # print("multi_preds: ", multi_preds)
            mean_preds_model = jnp.mean(jnp.array(multi_preds), axis=0)
            
            # batch_preds = jax.vmap(self._network.apply)(self._network_params, batch_genotypes)
            preds_model.append(mean_preds_model)

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

    # Train the model 
    @partial(jax.jit, static_argnames=("self", "num_epochs", "batch_size", "reevaluations", "l2_rate")) # OK
    def train_model_v2_unchanged(
        self,
        model_dict,
        training_set: Tuple[Genotype, Fitness, Descriptor],
        num_epochs: int = 300,
        batch_size: int = 512,
        reevaluations: int = 8,
        l2_rate: float = 0.000001,
        loss_fn = loss
    ):
        # Training set
        genotypes_train = training_set[0]
        fitness_train = training_set[1].reshape((-1, 1))
        descriptors_train = training_set[2]

        # Number of elements in training and validation sets
        num_train_samples = genotypes_train.shape[0]

        # Number of complete loops with training set
        num_batches_train = num_train_samples // batch_size
        # print(f"looping over {num_batches_train} batches from the buffer of size {num_train_samples}")

        # Keeping track of the training loss
        avg_train_losses = []
        avg_mse_losses = []

        # ==> SCAN pour les batchs mais garder boucle for pour epoch
        for epoch in tqdm.tqdm(range(num_epochs)):
        # for epoch in range(num_epochs):
            train_losses = []
            mse_losses = []
            # print(f"Epoch: {epoch}/{num_epochs}")

            for batch_num in range(num_batches_train):
                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size
                indices = jnp.arange(start_idx, end_idx)

                batch_genotypes = genotypes_train[indices]
                batch_fitness = fitness_train[indices]
                batch_descriptors = descriptors_train[indices]

                function_predictions = jnp.concatenate((batch_fitness, batch_descriptors), axis=1)

                # Working
                loss_fn_partial = functools.partial(loss_fn, model_dict['network_params'])
                loss_val, grads = jax.value_and_grad(loss_fn_partial, has_aux=True)(
                    model_dict['network_params'], self._network, batch_genotypes, function_predictions, l2_rate, reevaluations
                )

                # Update optimizer state
                updates, model_dict['optimizer'] = self._network_optimizer.update(grads, model_dict['optimizer'], model_dict['network_params'])

                # Update network parameters
                model_dict['network_params'] = optax.apply_updates(model_dict['network_params'], updates)

                train_losses.append(loss_val[0])
                mse_losses.append(loss_val[1][1])

            mse_losses = jnp.array(mse_losses, dtype=jnp.float32)

            avg_train_loss = jnp.mean(jnp.array(train_losses))
            avg_mse_loss = jnp.mean(mse_losses)

            avg_train_losses.append(avg_train_loss)
            array_avg_train_losses = jnp.array(avg_train_losses)
            avg_mse_losses.append(avg_mse_loss)
            array_avg_mse_losses = jnp.array(avg_mse_losses)

        return array_avg_train_losses, array_avg_mse_losses, model_dict




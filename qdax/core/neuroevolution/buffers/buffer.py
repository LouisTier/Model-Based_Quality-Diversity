"""  
This code is based on QDax framework: https://github.com/adaptive-intelligent-robotics/QDax/blob/main/qdax/core/neuroevolution/buffers/buffer.py
@article{lim2022accelerated,
  title={Accelerated Quality-Diversity for Robotics through Massive Parallelism},
  author={Lim, Bryan and Allard, Maxime and Grillotti, Luca and Cully, Antoine},
  journal={arXiv preprint arXiv:2202.01258},
  year={2022}
}

This work has been adapted and modified by Louis BERTHIER as part of his Individual Research Project at Imperial College London.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import flax
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, lax

from qdax.types import Action, Done, Observation, Reward, RNGKey, StateDescriptor, Descriptor, Fitness, Genotype


class Transition(flax.struct.PyTreeNode):
    """Stores data corresponding to a transition collected by a classic RL algorithm."""

    obs: Observation
    next_obs: Observation
    rewards: Reward
    dones: Done
    truncations: jnp.ndarray  # Indicates if an episode has reached max time step
    actions: Action

    @property
    def observation_dim(self) -> int:
        """
        Returns:
            the dimension of the observation
        """
        return self.obs.shape[-1]  # type: ignore

    @property
    def action_dim(self) -> int:
        """
        Returns:
            the dimension of the action
        """
        return self.actions.shape[-1]  # type: ignore

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.

        """
        flatten_dim = 2 * self.observation_dim + self.action_dim + 3
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: Transition,
    ) -> Transition:
        """
        Creates a transition from a flattened transition in a jnp.ndarray.

        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right

        Returns:
            a Transition object
        """
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim
        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim : (2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim) : (2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1) : (2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2) : (2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3) : (2 * obs_dim + 3 + action_dim)
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
        )

    @classmethod
    def init_dummy(cls, observation_dim: int, action_dim: int) -> Transition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = Transition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
        )
        return dummy_transition


class QDTransition(Transition):
    """Stores data corresponding to a transition collected by a QD algorithm."""

    state_desc: StateDescriptor
    next_state_desc: StateDescriptor

    @property
    def state_descriptor_dim(self) -> int:
        """
        Returns:
            the dimension of the state descriptors.

        """
        return self.state_desc.shape[-1]  # type: ignore

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.

        """
        flatten_dim = (
            2 * self.observation_dim
            + self.action_dim
            + 3
            + 2 * self.state_descriptor_dim
        )
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
                self.state_desc,
                self.next_state_desc,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: QDTransition,
    ) -> QDTransition:
        """
        Creates a transition from a flattened transition in a jnp.ndarray.

        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right

        Returns:
            a Transition object
        """
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim
        desc_dim = transition.state_descriptor_dim

        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim : (2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim) : (2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1) : (2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2) : (2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3) : (2 * obs_dim + 3 + action_dim)
        ]
        state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim) : (2 * obs_dim + 3 + action_dim + desc_dim),
        ]
        next_state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * desc_dim
            ),
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
        )

    @classmethod
    def init_dummy(  # type: ignore
        cls, observation_dim: int, action_dim: int, descriptor_dim: int
    ) -> QDTransition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.

        Args:
            observation_dim: observation dimension
            action_dim: action dimension

        Returns:
            a dummy transition
        """
        dummy_transition = QDTransition(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
            state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            next_state_desc=jnp.zeros(shape=(1, descriptor_dim)),
        )
        return dummy_transition


class ReplayBuffer(flax.struct.PyTreeNode):
    """
    A replay buffer where transitions are flattened before being stored.
    Transitions are unflatenned on the fly when sampled in the buffer.
    data shape: (buffer_size, transition_concat_shape)
    """

    data: jnp.ndarray
    buffer_size: int = flax.struct.field(pytree_node=False)
    transition: Transition

    current_position: jnp.ndarray = flax.struct.field()
    current_size: jnp.ndarray = flax.struct.field()

    @classmethod
    def init(
        cls,
        buffer_size: int,
        transition: Transition,
    ) -> ReplayBuffer:
        """
        The constructor of the buffer.

        Note: We have to define a classmethod instead of just doing it in post_init
        because post_init is called every time the dataclass is tree_mapped. This is a
        workaround proposed in https://github.com/google/flax/issues/1628.

        Args:
            buffer_size: the size of the replay buffer, e.g. 1e6
            transition: a transition object (might be a dummy one) to get
                the dimensions right
        """
        flatten_dim = transition.flatten_dim
        data = jnp.ones((buffer_size, flatten_dim)) * jnp.nan
        current_size = jnp.array(0, dtype=int)
        current_position = jnp.array(0, dtype=int)
        return cls(
            data=data,
            current_size=current_size,
            current_position=current_position,
            buffer_size=buffer_size,
            transition=transition,
        )

    @partial(jax.jit, static_argnames=("sample_size",))
    def sample(
        self,
        random_key: RNGKey,
        sample_size: int,
    ) -> Tuple[Transition, RNGKey]:
        """
        Sample a batch of transitions in the replay buffer.
        """
        random_key, subkey = jax.random.split(random_key)
        idx = jax.random.randint(
            subkey,
            shape=(sample_size,),
            minval=0,
            maxval=self.current_size,
        )
        samples = jnp.take(self.data, idx, axis=0, mode="clip")
        transitions = self.transition.__class__.from_flatten(samples, self.transition)
        return transitions, random_key

    @jax.jit
    def insert(self, transitions: Transition) -> ReplayBuffer:
        """
        Insert a batch of transitions in the replay buffer. The transitions are
        flattened before insertion.

        Args:
            transitions: A transition object in which each field is assumed to have
                a shape (batch_size, field_dim).
        """
        flattened_transitions = transitions.flatten()
        flattened_transitions = flattened_transitions.reshape(
            (-1, flattened_transitions.shape[-1])
        )
        num_transitions = flattened_transitions.shape[0]
        max_replay_size = self.buffer_size

        # Make sure update is not larger than the maximum replay size.
        if num_transitions > max_replay_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay "
                f"size. num_samples: {num_transitions}, "
                f"max replay size {max_replay_size}"
            )

        # get current position
        position = self.current_position

        # check if there is an overlap
        roll = jnp.minimum(0, max_replay_size - position - num_transitions)

        # roll the data to avoid overlap
        data = jnp.roll(self.data, roll, axis=0)

        # update the position accordingly
        new_position = position + roll

        # replace old data by the new one
        new_data = jax.lax.dynamic_update_slice_in_dim(
            data,
            flattened_transitions,
            start_index=new_position,
            axis=0,
        )

        # update the position and the size
        new_position = (new_position + num_transitions) % max_replay_size
        new_size = jnp.minimum(self.current_size + num_transitions, max_replay_size)

        # update the replay buffer
        replay_buffer = self.replace(
            current_position=new_position,
            current_size=new_size,
            data=new_data,
        )

        return replay_buffer  # type: ignore


"""
Contribution of Louis BERTHIER as part of his Individual Research Project.
"""

class TransitionBuffer:
    """Dataset storing genotypes and associated scores from the scoring function.

    Args:
        buffer_size: maximal size of the buffer to store elements
    """

    def __init__(
        self, 
        buffer_size: int
    ):
        
        self.buffer_size = buffer_size

    def add_transitions_explicit(
        self, 
        buffer,
        fitnesses_var,
        descriptors_var,
        genotypes: Genotype,
        fitnesses: Fitness, 
        descriptors: Descriptor
    ):
        """
        Add evaluations from the scoring function in the dataset with a FIFO implementation

        Args:
            genotypes: batch of solutions to be added
            fitnesses: batch of associated fitness score
            descriptors: batch of associated BD score
            fitnesses_var: batch of associated fitness uncertainty score
            descriptors_var: batch of associated BD uncertainty score

        Returns:
            nothing
        """
        
        batch_transition = jnp.hstack((genotypes, fitnesses.reshape(-1,1), descriptors, fitnesses_var.reshape(-1,1), descriptors_var.reshape(-1,1)))

        # Initialization
        if buffer is None:
            buffer = batch_transition

        # Next iterations
        else:
            # If not full, add new transitions
            if buffer.shape[0] < self.buffer_size:
                buffer = jnp.concatenate((buffer, batch_transition), axis=0)
            # If full, remove first elements and add new transitions
            else:
                buffer = buffer[genotypes.shape[0]:,:] # Remove the batch_size first elements
                buffer = jnp.concatenate((buffer, batch_transition), axis=0)
        
        return buffer

    def add_transitions_implicit(
        self, 
        buffer,
        genotypes: Genotype,
        fitnesses: Fitness, 
        descriptors: Descriptor
    ):
        """
        Add evaluations from the scoring function in the dataset with a FIFO implementation without uncertainty.

        Args:
            genotypes: batch of solutions to be added
            fitnesses: batch of associated fitness score
            descriptors: batch of associated BD score
            
        Returns:
            nothing
        """
        
        batch_transition = jnp.hstack((genotypes, fitnesses.reshape(-1,1), descriptors))

        # Initialization
        if buffer is None:
            buffer = batch_transition

        # Next iterations
        else:
            # If not full, add new transitions
            if buffer.shape[0] < self.buffer_size:
                buffer = jnp.concatenate((buffer, batch_transition), axis=0)
            # If full, remove first elements and add new transitions
            else:
                buffer = buffer[genotypes.shape[0]:,:] # Remove the batch_size first elements
                buffer = jnp.concatenate((buffer, batch_transition), axis=0)

        return buffer

    def split_buffer(
        self,
        buffer,
        random_key: RNGKey,
        train_ratio: float = 0.7,
    ) -> Tuple[Tuple[Genotype, Fitness, Descriptor], Tuple[Genotype, Fitness, Descriptor], RNGKey]: 
        """
        Split buffer into a training and testing dataset.

        Args:
            random_key: a JAX PRNG key
            train_ratio: percentage of the data considered for the training dataset
            
        Returns:
            training dataset with genotypes and scores
            testing dataset with genotypes and scores
            a new JAX PRNG key
        """

        # Retrieve elements from buffer
        genotypes = buffer[:,:8]
        fitnesses = buffer[:,8:9]
        descriptors = buffer[:,9:11]

        # Shuffle indices (randomize individuals)
        random_key, subkey = jax.random.split(random_key)
        num_samples = genotypes.shape[0]
        shuffled_indices = jax.random.permutation(subkey, num_samples)

        # Calculate the split point to dissociate train and validation sets
        split_point = int(num_samples * train_ratio)
        train_indices = shuffled_indices[:split_point]
        val_indices = shuffled_indices[split_point:]

        # Training set
        genotypes_train = genotypes[train_indices]
        fitness_train = fitnesses[train_indices]
        descriptors_train = descriptors[train_indices]

        training_set = [genotypes_train, fitness_train, descriptors_train]

        # Validation set
        genotypes_val = genotypes[val_indices]
        fitness_val = fitnesses[val_indices]
        descriptors_val = descriptors[val_indices]

        testing_set = [genotypes_val, fitness_val, descriptors_val]

        return training_set, testing_set, random_key

class TransitionBuffer_UQ:
    """Dataset storing genotypes and associated scores and uncertainty from the scoring function.

    Args:
        buffer_size: maximal size of the buffer to store elements
    """

    def __init__(
        self, 
        buffer_size: int
    ):
        
        self.buffer_size = buffer_size

    def z_score_normalisation(
        self, 
        x
    ):
        """
        Normalize the input.

        Args:
            x: An element (usually scores of the solutions)  
           
        Returns:
            Normalized input
            Mean parameter used for normalization
            Std parameter used for normalization
        """

        mean = jnp.mean(x)
        std = jnp.std(x)
        z_score = (x - mean) / std

        return z_score, mean, std
        
    def normalize_buffer_explicit(
        self,
        buffer,
    ):
        
        genotypes = buffer[:,:8]
        fitnesses = buffer[:,8:9]
        bd1 = buffer[:,9:10]
        bd2 = buffer[:,10:11]
        fitnesses_var = buffer[:,11:12]
        descriptors_var = buffer[:,12:13]

        norm_fitnesses, mean_fitnesses, std_fitnesses = self.z_score_normalisation(x=fitnesses)
        norm_bd1, mean_bd1, std_bd1 = self.z_score_normalisation(x=bd1)
        norm_bd2, mean_bd2, std_bd2 = self.z_score_normalisation(x=bd2)
        norm_fitnesses_var, mean_fitnesses_var, std_fitnesses_var = self.z_score_normalisation(x=fitnesses_var)
        norm_descriptors_var, mean_descriptors_var, std_descriptors_var = self.z_score_normalisation(x=descriptors_var)

        norm_buffer = jnp.hstack((genotypes, norm_fitnesses.reshape(-1,1), norm_bd1.reshape(-1,1), norm_bd2.reshape(-1,1), norm_fitnesses_var.reshape(-1,1), norm_descriptors_var.reshape(-1,1)))
        norm_param = [mean_fitnesses, std_fitnesses, mean_bd1, std_bd1, mean_bd2, std_bd2, mean_fitnesses_var, std_fitnesses_var, mean_descriptors_var, std_descriptors_var]

        return norm_buffer, norm_param, buffer

    def add_transitions_explicit(
        self, 
        buffer,
        fitnesses_var,
        descriptors_var,
        genotypes: Genotype,
        fitnesses: Fitness, 
        bd1: Descriptor,
        bd2: Descriptor
    ):
        """
        Add evaluations from the scoring function in the dataset with a FIFO implementation

        Args:
            genotypes: batch of solutions to be added
            fitnesses: batch of associated fitness score
            descriptors: batch of associated BD score
            fitnesses_var: batch of associated fitness uncertainty score
            descriptors_var: batch of associated BD uncertainty score

        Returns:
            nothing
        """
        
        # batch_transition = jnp.hstack((genotypes, fitnesses.reshape(-1,1), descriptors, fitnesses_var.reshape(-1,1), descriptors_var.reshape(-1,1)))
        batch_transition = jnp.hstack((genotypes, fitnesses.reshape(-1,1), bd1.reshape(-1,1), bd2.reshape(-1,1), fitnesses_var.reshape(-1,1), descriptors_var.reshape(-1,1)))

        # Initialization
        if buffer is None:
            buffer = batch_transition

        # Next iterations
        else:
            # If not full, add new transitions
            if buffer.shape[0] < self.buffer_size:
                buffer = jnp.concatenate((buffer, batch_transition), axis=0)
            # If full, remove first elements and add new transitions
            else:
                buffer = buffer[genotypes.shape[0]:,:] # Remove the batch_size first elements
                buffer = jnp.concatenate((buffer, batch_transition), axis=0)
        
        return buffer
    
    def normalize_buffer_implicit(
        self,
        buffer,
    ):
        
        genotypes = buffer[:,:8]
        fitnesses = buffer[:,8:9]
        bd1 = buffer[:,9:10]
        bd2 = buffer[:,10:11]

        norm_fitnesses, mean_fitnesses, std_fitnesses = self.z_score_normalisation(x=fitnesses)
        norm_bd1, mean_bd1, std_bd1 = self.z_score_normalisation(x=bd1)
        norm_bd2, mean_bd2, std_bd2 = self.z_score_normalisation(x=bd2)

        norm_buffer = jnp.hstack((genotypes, norm_fitnesses.reshape(-1,1), norm_bd1.reshape(-1,1), norm_bd2.reshape(-1,1)))
        norm_param = [mean_fitnesses, std_fitnesses, mean_bd1, std_bd1, mean_bd2, std_bd2]

        return norm_buffer, norm_param, buffer

    def add_transitions_implicit(
        self, 
        buffer,
        genotypes: Genotype,
        fitnesses: Fitness, 
        bd1: Descriptor,
        bd2: Descriptor
    ):
        """
        Add evaluations from the scoring function in the dataset with a FIFO implementation without uncertainty.

        Args:
            genotypes: batch of solutions to be added
            fitnesses: batch of associated fitness score
            descriptors: batch of associated BD score
            
        Returns:
            nothing
        """

        batch_transition = jnp.hstack((genotypes, fitnesses.reshape(-1,1), bd1.reshape(-1,1), bd2.reshape(-1,1)))

        # Initialization
        if buffer is None:
            buffer = batch_transition

        # Next iterations
        else:
            # If not full, add new transitions
            if buffer.shape[0] < self.buffer_size:
                buffer = jnp.concatenate((buffer, batch_transition), axis=0)
            # If full, remove first elements and add new transitions
            else:
                buffer = buffer[genotypes.shape[0]:,:] # Remove the batch_size first elements
                buffer = jnp.concatenate((buffer, batch_transition), axis=0)
                
        return buffer

    def split_buffer(
        self,
        buffer,
        random_key: RNGKey,
        train_ratio: float = 0.7,
    ) -> Tuple[Tuple[Genotype, Fitness, Descriptor, Descriptor, Fitness, Descriptor], Tuple[Genotype, Fitness, Descriptor, Descriptor, Fitness, Descriptor], RNGKey]: 
        """
        Split buffer into a training and testing dataset with uncertainty.

        Args:
            random_key: a JAX PRNG key
            train_ratio: percentage of the data considered for the training dataset
            
        Returns:
            training dataset with genotypes and scores + uncertainty
            testing dataset with genotypes and scores + uncertainty
            a new JAX PRNG key
        """

        # Retrieve elements from buffer
        genotypes = buffer[:,:8]
        fitnesses = buffer[:,8:9]
        bd1 = buffer[:,9:10]
        bd2 = buffer[:,10:11]
        fitnesses_var = buffer[:,11:12]
        descriptors_var = buffer[:,12:13] 

        # Shuffle indices (randomize individuals)
        random_key, subkey = jax.random.split(random_key)
        num_samples = genotypes.shape[0]
        shuffled_indices = jax.random.permutation(subkey, num_samples)

        # Calculate the split point to dissociate train and validation sets
        split_point = int(num_samples * train_ratio)
        train_indices = shuffled_indices[:split_point]
        val_indices = shuffled_indices[split_point:]

        # Training set
        genotypes_train = genotypes[train_indices]
        fitness_train = fitnesses[train_indices]
        bd1_train = bd1[train_indices]
        bd2_train = bd2[train_indices]
        fitness_var_train = fitnesses_var[train_indices]
        descriptors_var_train = descriptors_var[train_indices]

        training_set = [genotypes_train, fitness_train, bd1_train, bd2_train, fitness_var_train, descriptors_var_train]

        # Testing set
        genotypes_val = genotypes[val_indices]
        fitness_val = fitnesses[val_indices]
        bd1_val = bd1[val_indices]
        bd2_val = bd2[val_indices]
        fitness_var_val = fitnesses_var[val_indices]
        descriptors_var_val = descriptors_var[val_indices]

        testing_set = [genotypes_val, fitness_val, bd1_val, bd2_val, fitness_var_val, descriptors_var_val]

        return training_set, testing_set, random_key

# class TransitionBuffer:
#     """Dataset storing genotypes and associated scores from the scoring function.

#     Args:
#         buffer_size: maximal size of the buffer to store elements
#     """

#     def __init__(
#         self, 
#         buffer_size: int
#     ):
        
#         self.buffer_size = buffer_size
#         self.buffer = None # jnp.empty((buffer_size,), dtype=jnp.float32) # jnp.empty((buffer_size,), dtype=jnp.float32) or None

#     def add_transitions_2(
#         self, 
#         fitnesses_var,
#         descriptors_var,
#         genotypes: Genotype,
#         fitnesses: Fitness, 
#         descriptors: Descriptor
#     ):
#         """
#         Add evaluations from the scoring function in the dataset with a FIFO implementation

#         Args:
#             genotypes: batch of solutions to be added
#             fitnesses: batch of associated fitness score
#             descriptors: batch of associated BD score
#             fitnesses_var: batch of associated fitness uncertainty score
#             descriptors_var: batch of associated BD uncertainty score

#         Returns:
#             nothing
#         """
        
#         batch_transition = jnp.hstack((genotypes, fitnesses.reshape(-1,1), descriptors, fitnesses_var.reshape(-1,1), descriptors_var.reshape(-1,1)))

#         # Initialization
#         if self.buffer is None:
#         # if self.buffer.size == 0:
#             self.buffer = batch_transition

#         # Next iterations
#         else:
#             # If not full, add new transitions
#             if self.buffer.shape[0] < self.buffer_size:
#                 self.buffer = jnp.concatenate((self.buffer, batch_transition), axis=0)
#             # If full, remove first elements and add new transitions
#             else:
#                 self.buffer = self.buffer[genotypes.shape[0]:,:] # Remove the batch_size first elements
#                 self.buffer = jnp.concatenate((self.buffer, batch_transition), axis=0)

#     def add_transitions_implicit(
#         self, 
#         genotypes: Genotype,
#         fitnesses: Fitness, 
#         descriptors: Descriptor
#     ):
#         """
#         Add evaluations from the scoring function in the dataset with a FIFO implementation without uncertainty.

#         Args:
#             genotypes: batch of solutions to be added
#             fitnesses: batch of associated fitness score
#             descriptors: batch of associated BD score
            
#         Returns:
#             nothing
#         """
        
#         batch_transition = jnp.hstack((genotypes, fitnesses.reshape(-1,1), descriptors))

#         # Initialization
#         if self.buffer is None:
#         # if self.buffer.size == 0:
#             self.buffer = batch_transition

#         # Next iterations
#         else:
#             # If not full, add new transitions
#             if self.buffer.shape[0] < self.buffer_size:
#                 self.buffer = jnp.concatenate((self.buffer, batch_transition), axis=0)
#             # If full, remove first elements and add new transitions
#             else:
#                 self.buffer = self.buffer[genotypes.shape[0]:,:] # Remove the batch_size first elements
#                 self.buffer = jnp.concatenate((self.buffer, batch_transition), axis=0)

#     def split_buffer(
#             self,
#             random_key: RNGKey,
#             train_ratio: float = 0.7,
#     ) -> Tuple[Tuple[Genotype, Fitness, Descriptor], Tuple[Genotype, Fitness, Descriptor], RNGKey]: 
#         """
#         Split buffer into a training and testing dataset.

#         Args:
#             random_key: a JAX PRNG key
#             train_ratio: percentage of the data considered for the training dataset
            
#         Returns:
#             training dataset with genotypes and scores
#             testing dataset with genotypes and scores
#             a new JAX PRNG key
#         """

#         # Retrieve elements from buffer
#         genotypes = self.buffer[:,:8]
#         fitnesses = self.buffer[:,8:9]
#         descriptors = self.buffer[:,9:11]

#         # Shuffle indices (randomize individuals)
#         random_key, subkey = jax.random.split(random_key)
#         num_samples = genotypes.shape[0]
#         shuffled_indices = jax.random.permutation(subkey, num_samples)

#         # Calculate the split point to dissociate train and validation sets
#         split_point = int(num_samples * train_ratio)
#         # split_point = jnp.floor(num_samples * train_ratio).astype(int)
#         train_indices = shuffled_indices[:split_point]
#         val_indices = shuffled_indices[split_point:]

#         # Training set
#         genotypes_train = genotypes[train_indices]
#         fitness_train = fitnesses[train_indices]
#         descriptors_train = descriptors[train_indices]

#         # Validation set
#         genotypes_val = genotypes[val_indices]
#         fitness_val = fitnesses[val_indices]
#         descriptors_val = descriptors[val_indices]

#         return [genotypes_train, fitness_train, descriptors_train], [genotypes_val, fitness_val, descriptors_val], random_key

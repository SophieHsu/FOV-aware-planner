import akro
import torch
from torch import nn

from garage import InOutSpec
from garage.torch.modules import GaussianMLPModule, CNNModule
from garage.torch.value_functions.value_function import ValueFunction


class GaussianCNNMLPValueFunction(ValueFunction):
    """Gaussian CNN value function.


    It fits CNN input to gaussian distribution estimated by a MLP.

    Args:
        env_spec (EnvSpec): Environment specification.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match env_spec. Gym
            uses NHWC by default, but PyTorch uses NCHW by default.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_channels (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        paddings (tuple[int]):  Zero-padding added to both sides of the input
        padding_mode (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): The name of the value function.
    """
    def __init__(self,
                 env_spec,
                 image_format,
                 kernel_sizes,
                 *,
                 hidden_channels,
                 strides=1,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 paddings=0,
                 padding_mode='zeros',
                 max_pool=False,
                 pool_shape=None,
                 pool_stride=1,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 layer_normalization=False,
                 name='GaussianCNNMLPValueFunction'):
        if isinstance(env_spec.observation_space, akro.Dict):
            raise ValueError('CNN policies do not support '
                             'with akro.Dict observation spaces.')
        self._env_spec = env_spec

        super(GaussianCNNMLPValueFunction, self).__init__(env_spec, name)

        self._cnn_module = CNNModule(InOutSpec(
            self._env_spec.observation_space, None),
                                     image_format=image_format,
                                     kernel_sizes=kernel_sizes,
                                     strides=strides,
                                     hidden_channels=hidden_channels,
                                     hidden_w_init=hidden_w_init,
                                     hidden_b_init=hidden_b_init,
                                     hidden_nonlinearity=hidden_nonlinearity,
                                     paddings=paddings,
                                     padding_mode=padding_mode,
                                     max_pool=max_pool,
                                     pool_shape=pool_shape,
                                     pool_stride=pool_stride,
                                     layer_normalization=layer_normalization)
        self._gaussian_mlp_module = GaussianMLPModule(
            input_dim=self._cnn_module.spec.output_space.flat_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=None,
            max_std=None,
            std_parameterization='exp',
            layer_normalization=layer_normalization)

    def compute_loss(self, obs, returns):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
        observations = obs.reshape(-1, *self._env_spec.observation_space.shape)
        cnn_output = self._cnn_module(observations)
        dist = self._gaussian_mlp_module(cnn_output)

        ll = dist.log_prob(returns.reshape(-1, 1))
        loss = -ll.mean()
        return loss

    # pylint: disable=arguments-differ
    def forward(self, obs):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """
        # In case we're given flattened observations.
        non_space_dims = obs.shape[:len(obs.shape) -
                                   len(self._env_spec.observation_space.shape)]
        observations = obs.reshape(-1, *self._env_spec.observation_space.shape)
        cnn_output = self._cnn_module(observations)
        cnn_output = cnn_output.reshape(
            *non_space_dims, *self._cnn_module.spec.output_space.shape)
        dist = self._gaussian_mlp_module(cnn_output)

        return dist.mean.flatten(-2)
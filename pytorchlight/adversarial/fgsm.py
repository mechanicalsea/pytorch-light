import pytorchlight.utils.log as L
import torch
from .frame import White

fgsm_logger = L.Log('fgsm')


class FGSM(White):
    """Fast Gradient Sign Method for adversarial examples.

    This simple and fast method of generating adversarial examples comes from 
    a view of neural networks' vulnerability to adversarial perturbation is 
    their linear nature. The view is identified by Goodfellow et. al. in 
    [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572).
    """

    def __init__(self, model, **params):
        """Define Fast Gradient Sign Method.

        Generating a FGSM attacker.

        Args:
            model (Model): target object
            **params (dict): a series of pairs of FGSM's parameters

        Example::

            >>> import torch.nn.functional as F
            >>> model = Mnist_CNN()
            >>> fgsm = FGSM(model, loss=F.nll_loss, epsilon=0.0)

        """
        super(FGSM, self).__init__(model)
        self.loss = params.get('loss', None)
        self.epsilon = params.get('epsilon', 0.0)
        if self.loss is None:
            fgsm_logger.error(
                'ValueError - Missing loss function: {loss_function}, \
                please then set loss'.format(self.loss))
            raise ValueError(
                'Missing loss function: {loss_function}, \
                please then set loss'.format(self.loss))

    def adv(self, x):
        """Generate adversarial examples based on `x` through FGSM.

        input -> perturbated input: $perturbed\_x = x + \epsilon*sign(x\_grad)$

        Args:
            **input (dict of tensor): A FGSM input indicates an attack object 
            which can be features of various forms such as image(s), voice(s), 
            and one-verctor number(s), and object gradient which is 
            corresponding to the object. 
            **Note that the pair of x and gradient should be same shape.**

        Example::

            >>> import torch
            >>> import torch.nn.functional as F
            >>> model = Mnist_CNN()
            >>> fgsm = FGSM(model, loss=F.nll_loss, epsilon=0.0)
            >>> perturbed_x = fgsm(x=torch.tensor([[1, 2, 3]]))
            >>> perturbed_x = torch.clamp(perturbed_x, 0, 1)

        Returns:
            (tensor): perturbed x. If x is image, x should be then clipped, 
            e.g., by using torch.clamp(x, min=0, max=1)
        """
        if not isinstance(x, torch.Tensor):
            fgsm_logger.error(
                'TypeError - Invalid Type of x: {}'.format(
                    type(x).__name__))
            raise TypeError(
                'TypeError - Invalid Type of x: {}'.format(
                    type(x).__name__))
        if x is None:
            fgsm_logger.error(
                'ValueError - Excepted x: {}'.format(None))
            raise ValueError(
                'ValueError - Excepted x: {}'.format(None))
        if x.requires_grad is False:
            x.requires_grad = True
        # Calculate gradients of model in terms of x
        output = self.model(x)
        pred_y = output.max(1)[1]
        loss = self.loss(output, pred_y)
        self.model.zero_grad()
        loss.backward()
        grad = x.grad.data
        # Call FGSM Attack
        x_adv = x + self.epsilon * grad.sign()
        return x_adv

    def set_params(self, **params):
        """Reset parameters of Fast Gradient Sign Method (FGSM).

        Update epsilon.

        Args:
            **params (dict): FGSM's parameters, such as epsilon.

        Example::

            >>> fgsm = FGSM(epsilon=0.1)
            >>> fgsm.set_params(epsilon=0.25)

        """
        self.epsilon = params.get('epsilon', self.epsilon)

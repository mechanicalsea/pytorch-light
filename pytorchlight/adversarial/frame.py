import torch
from torch import nn
import torch.nn.functional as F


class BoxSetting(nn.Module):

    def __init__(self):
        """Construct box setting which is used to yield white-box and 
        black-box.

        Abstract of Box Setting provide the function to generate adversarial 
        examples. In this setting, model and parameters may be offered.
        """
        super(BoxSetting, self).__init__()

    def forward(self, x):
        """Using function `self.adv` to generate adversarial examples.

        This is an interface for creating adversarial examples, and, 
        without rewritting, it is directly used to generate examples.

        Args:
            x (Tensor): real features of data (input)

        Returns:
            (Tensor): adversarial examples which has same shape of input `x`
        """
        return self.adv(x)

    def adv(x):
        """Define the method of generating adversarial examples.

        The attack method is defined here which can an iteration or closed-form solution.
        Ultimately, this is contributed to `self.adv` which is the function to generate
        adversarial examples.
        **Note that, in each method, this function must be rewritten.**

        Args:
            x (Tensor): Features of single or batch data

        """
        return None

    def set_params(**params):
        """Interface for updating parameters.

        It's ok without any updates of parameters, and it is can be replaced by directly using
        `self.para_1 = value_1`.

        Args:
            **params (dict): parameters of attack of test
        """
        pass


class White(BoxSetting):

    def __init__(self, model):
        """White-box setting for tester.

        In the white box scenario, the adversary has full knowledge of the model 
        including model type, model architecture and values of all parameters and 
        trainable weights. So the model is required.

        Args:
            model (Model): target object
            **params (dict): parameters of attack or test, such as {'loss': loss_func}
        """
        super(White, self).__init__()
        self.model = model


class Black(BoxSetting):

    def __init__(self, model=None):
        """Black-box setting for tester.

        In this scenario, the adversary does not know very much about the model. 
        There are also two forms of the Black based on whether with probing or 
        not. Black box With probing admits to probe or query the model, i.e. feed 
        some inputs and observe outputs. So, if probing is allowed, the target 
        model will occur in the parametes, e.g. {'target': Model_1}

        Args:
            model (Model): Model for generating adversarial examples that may not be provided 
                           (default: {None})
            **params: the parameter of method, such as {'target': Model_1}

        """
        super(Black, self).__init__()
        self.model = model


def test(model, dataloader, adv, device=torch.device('cpu')):
    """Experiment for adversarial attack.

    Given data and attack's parameters are conducted to quantify performance.

    Args:
        dataloader (DataLoader): data loader of test set which includes 
                                 features and labels, and the batch size 
                                 must be 1
        device (device): torch.device('cpu') or torch.device('cuda') 
                         (default: {torch.device('cpu')})
        **params (dict): attack's parameters

    Example::
        
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>> dataloader = DataLoader(TensorDataset(torch.Tensor([[1, 2, 3],
                                                                [4, 5, 6]])
                                                  torch.Tensor([0, 1])),
                                    shuffle=True, batch_size=1)
        >>> model = Model()  # Define a model for classification
        >>> fgsm = FGSM(model, loss=F.nll_loss, epsilon=0.05)
        >>> experiment(model, dataloader, fgsm, device=torch.device('cpu'))

    Returns:
        (float): accuracy with adversarial examples

    """
    # Accuracy counter
    correct = 0
    # Loop over all examples in test set
    for x, y in dataloader:
        # Send the data and label to the device
        x, y = x.to(device), y.to(device)
        # Forward pass the data through the model
        output = model(x)
        # get the index of the max log-probability
        pred_x = output.max(1, keepdim=True)[1]
        # It's meaningless if the initial prediction is wrong
        if pred_x.item() != y.item():
            continue
        # generate adversarial example and clip it
        perturbed_x = adv(x)
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        # Re-classify the perturbed image
        output = model(perturbed_x)
        # get the index of the max log-probability
        pred_adv = output.max(1, keepdim=True)[1]
        # Check for success
        if pred_adv.item() == y.item():
            correct += 1
    # Calculate final accuracy
    final_acc = correct / float(len(dataloader))
    print("Test Accuracy = {} / {} = {}".format(correct,
                                                len(dataloader), final_acc))
    # Return the accuracy and an adversarial example
    return final_acc

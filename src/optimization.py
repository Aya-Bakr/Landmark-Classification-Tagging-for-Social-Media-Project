import torch
import torch.nn as nn
import torch.optim

def get_loss():
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True
    """
    loss = nn.CrossEntropyLoss()
    return loss

def get_optimizer(
    model: nn.Module,
    optimizer: str = "Adam",
    learning_rate: float = 0.0001,
    momentum: float = 0.9,
    weight_decay: float = 0.001,
):
    """
    Returns an optimizer instance

    :param model: the model to optimize
    :param optimizer: one of 'SGD' or 'Adam'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    """
    if optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt

def get_scheduler(optimizer, scheduler: str = "StepLR", step_size: int = 10, gamma: float = 0.1):
    """
    Returns a learning rate scheduler instance

    :param optimizer: the optimizer instance to attach the scheduler to
    :param scheduler: one of 'StepLR', 'ExponentialLR', 'ReduceLROnPlateau'
    :param step_size: step size for 'StepLR'
    :param gamma: multiplicative factor of learning rate decay
    """
    if scheduler.lower() == "steplr":
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler.lower() == "exponentiallr":
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler.lower() == "reducelronplateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        raise ValueError(f"Scheduler {scheduler} not supported")

    return sched

######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)

def test_get_loss():
    loss = get_loss()
    assert isinstance(
        loss, nn.CrossEntropyLoss
    ), f"Expected cross entropy loss, found {type(loss)}"

def test_get_optimizer_type(fake_model):
    opt = get_optimizer(fake_model)
    assert isinstance(opt, torch.optim.Adam), f"Expected Adam optimizer, got {type(opt)}"

def test_get_optimizer_is_linked_with_model(fake_model):
    opt = get_optimizer(fake_model)
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])

def test_get_optimizer_returns_adam(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam")
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
    assert isinstance(opt, torch.optim.Adam), f"Expected Adam optimizer, got {type(opt)}"

def test_get_optimizer_sets_learning_rate(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123)
    assert (
        opt.param_groups[0]["lr"] == 0.123
    ), "get_optimizer is not setting the learning rate appropriately. Check your code."

def test_get_optimizer_sets_momentum(fake_model):
    opt = get_optimizer(fake_model, optimizer="SGD", momentum=0.123)
    assert (
        opt.param_groups[0]["momentum"] == 0.123
    ), "get_optimizer is not setting the momentum appropriately. Check your code."

def test_get_optimizer_sets_weight_decay(fake_model):
    opt = get_optimizer(fake_model, optimizer="SGD", weight_decay=0.123)
    assert (
        opt.param_groups[0]["weight_decay"] == 0.123
    ), "get_optimizer is not setting the weight_decay appropriately. Check your code."

def test_get_scheduler(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam")
    sched = get_scheduler(opt)
    assert isinstance(sched, torch.optim.lr_scheduler.StepLR), f"Expected StepLR scheduler, got {type(sched)}"
import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot

def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """
    if torch.cuda.is_available():
        # YOUR CODE HERE: transfer the model to the GPU
        # HINT: use .cuda()
        model = model.cuda()
    # YOUR CODE HERE: set the model to training mode

    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # 1. clear the gradients of all optimized variables
        # YOUR CODE HERE:
        optimizer.zero_grad()
        output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()

    train_loss /= len(train_dataloader)
    return train_loss

def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """
    valid_loss = 0.0  # Declare valid_loss

    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            model.cuda()

        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # 2. forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)
            loss_value = loss(output, target)
            valid_loss += loss_value.item()

    valid_loss /= len(valid_dataloader)
    return valid_loss

def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = float('inf')
    logs = {}
    
    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    # HINT: look here: 
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.01)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)
        
        # print training/validation statistics

        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}")

        if valid_loss < valid_loss_min:
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")
            
            # Save the weights to save_path
            # YOUR CODE HERE
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

         # Update learning rate, i.e., make a step in the learning rate scheduler
        # YOUR CODE HERE
        scheduler.step(valid_loss)
        
        # Log the losses and the current learning rate

        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()

def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy

    test_loss = 0.0
    correct = 0
    total = 0
    
    # set the module to evaluation mode

    with torch.no_grad():
        
        # set the model to evaluation mode
        # YOUR CODE HERE
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
            enumerate(test_dataloader),
            desc='Testing',
            total=len(test_dataloader),
            leave=True,
            ncols=80
        ):
            # move data to GPU

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
                
            # 1. forward pass: compute predicted outputs by passing inputs to the model

            logits = model(data)
            # 2. calculate the loss
            loss_value = loss(logits, target)
            # update average test loss
            test_loss += loss_value.item()
            
            # convert logits to predicted class
            # HINT: the predicted class is the index of the max of the logits

            pred = torch.argmax(logits, dim=-1)
            # compare predictions to true label
            correct += torch.sum(pred == target).item()
            total += target.size(0)

    test_loss /= len(test_dataloader)
    accuracy = 100. * correct / total

    print(f'Test Loss: {test_loss:.6f}')
    print(f'Test Accuracy: {accuracy:.2f}% ({correct}/{total})')

    return test_loss

######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)

@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel
    model = MyModel(50)
    return model, get_loss(), get_optimizer(model)

def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"

def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")

def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
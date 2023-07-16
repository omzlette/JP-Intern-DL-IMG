import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# For Training Function
from tqdm.auto import tqdm
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.\n
        Credit to: (https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)"""
    
    def __init__(self, patience=7, verbose=False, delta=0, path='models/earlystop/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists('models/earlystop/'):
            os.makedirs('models/earlystop/')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


#####################################################

def plot_images(data:list,
                datalabel:list,
                allLabel:list):
    """
    This function plots the images in a 5x5 grid.

    Args:
        data (list): List of images to plot.
        datalabel (list): List of labels for the images.
        allLabel (list): List of all the labels.
    """
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[i])
        plt.xlabel(allLabel[datalabel[i]])

#####################################################

def train_model(models:list,
                modelnames:list,
                criterion:torch.nn.modules.loss,
                optimizer:torch.optim,
                scheduler:torch.optim.lr_scheduler, 
                num_epochs:int, 
                train_loader:torch.utils.data.DataLoader,
                test_loader:torch.utils.data.DataLoader, 
                device:torch.device, 
                patience:int=3, 
                path_to_folder:str='models/earlystop', 
                verbose:bool=True) -> dict:

    """
    This function trains and tests the model(s) and returns the training and testing loss and accuracy of the model(s).

    Args:
        models (list): List of models to train and test.
        modelnames (list): List of model names.
        criterion (torch.nn.modules.loss): Loss function.
        optimizer (torch.optim): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train the model(s).
        train_loader (torch.utils.data.DataLoader): Training data loader.
        test_loader (torch.utils.data.DataLoader): Testing data loader.
        device (torch.device): Device to train the model(s) on.
        patience (int): Number of epochs to wait for improvement before early stopping.
                        Default: 3
        path_to_folder (str): Path to folder to save the model(s).
                        Default: 'models/earlystop'
        verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: True

    Returns:
        traintestData (dict): Dictionary containing the loss and accuracy of the model(s).
    """

    verboseprint = print if verbose else lambda *a, **k: None

    # Store the loss and accuracy
    traintestData = {}

    for model, modelname in zip(models, modelnames):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience, path=f'{path_to_folder}/{modelname}.pt')

        # Bar format
        bar_format = "{desc} |{bar}| {n_fmt}/{total_fmt} [Time Elapsed: {elapsed}/{remaining},{rate_fmt}]"

        for param in model.parameters():
            param.requires_grad_()
        
        with tqdm(total=num_epochs, desc=f'{modelname} ->',
                  unit='epoch',bar_format=bar_format, disable=not verbose) as pbar:
            for epoch in range(num_epochs):
                
                if early_stopping.early_stop:
                    verboseprint(f"Early stopping at epoch {epoch}")
                    break

                ### Train the model ###

                model.train()
                running_loss = 0.0
                train_correct = 0
                train_samples = 0
                
                with tqdm(enumerate(train_loader), desc=f'Training -> Epoch {epoch+1}/{num_epochs}',
                                        unit='batch', bar_format=bar_format + " {postfix}", disable=not verbose, total=len(train_loader)) as pbar2:
                    for _, batch in pbar2:
                        images = batch[0].to(device)
                        labels = batch[1].to(device).long()

                        optimizer.zero_grad()
                        
                        outputs = model(images)

                        # Convert outputs to tensor
                        if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                            outputs = outputs.logits

                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * images.size(0)

                        # Training Accuracy
                        _, predicted = torch.max(outputs, dim=1)
                        train_correct += (predicted == labels).sum().item()
                        train_samples += labels.size(0)

                        # Update progress bar
                        pbar2.set_postfix_str(f"Train Loss: {running_loss/len(train_loader.dataset):.4f}, Train Acc: {train_correct/train_samples:.4f}")

                train_epoch_loss = running_loss/len(train_loader.dataset)
                train_epoch_acc = train_correct/train_samples

                if scheduler is not None:
                    scheduler.step()

                ### Test the model ###

                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    test_correct = 0
                    test_total = 0
                    
                    with tqdm(enumerate(test_loader), desc=f'Testing -> Epoch {epoch+1}/{num_epochs}',
                                         unit='batch', bar_format=bar_format+" {postfix}", disable=not verbose, total=len(test_loader)) as pbar2:
                        for _, batch in pbar2:
                            images = batch[0].to(device)
                            labels = batch[1].to(device).long()

                            outputs = model(images)

                            if isinstance(outputs, ImageClassifierOutputWithNoAttention):
                                outputs = outputs.logits

                            # Test Loss
                            loss = criterion(outputs, labels)
                            total_loss += loss.item() * images.size(0)

                            # Test Accuracy
                            _, predicted = torch.max(outputs, dim=1)
                            test_total += labels.size(0)
                            test_correct += (predicted == labels).sum().item()

                            # Update progress bar
                            pbar2.set_postfix_str(f"Test Loss: {total_loss/len(test_loader.dataset):.4f}, Test Acc: {test_correct/test_total:.4f}")
                        
                        test_epoch_loss = total_loss/len(test_loader.dataset)
                        test_epoch_acc = test_correct/test_total

                # Save the loss and accuracy for train and test set
                train_loss.append(train_epoch_loss)
                train_acc.append(train_epoch_acc)
                test_loss.append(test_epoch_loss)
                test_acc.append(test_epoch_acc)

                # verboseprint(f'''{modelname} -> Epoch: {epoch+1}/{num_epochs} 
                # Train Loss: {train_epoch_loss} | Train Accuracy: {train_epoch_acc} 
                # Test Loss: {test_epoch_loss} | Test Accuracy: {test_epoch_acc}''')

                # Early Stopping
                early_stopping(test_epoch_loss, model)


                
                pbar.update()

        traintestData[f'Training Loss_{modelname}'] = train_loss
        traintestData[f'Training Accuracy_{modelname}'] = train_acc
        traintestData[f'Test Loss_{modelname}'] = test_loss
        traintestData[f'Test Accuracy_{modelname}'] = test_acc

    return traintestData

#####################################################

def lossAccPlot(lossAccDict:dict,
                modelnames:list):
    """
    Plot the loss and accuracy of the model(s).

    Args:
        lossAccDict (dict): Dictionary containing the loss and accuracy of the model(s).
        modelnames (list): List of model names.
    """

    names = ['Loss', 'Accuracy']

    num_models = len(modelnames)
    num_plots = len(names)

    _, axes = plt.subplots(num_models, num_plots, figsize=(20, 5*num_models))

    # If there is only one row or one column of subplots, convert axes to a 2-dimensional array
    if num_models == 1 or num_plots == 1:
        axes = axes.reshape(num_models, num_plots)

    for idx, modelname in enumerate(modelnames):
        for i, name in enumerate(names):
            val_train = lossAccDict[f'Training Loss_{modelname}'] if name == 'Loss' else lossAccDict[f'Training Accuracy_{modelname}']
            axes[idx, i].plot(val_train, label='Training')

            val_test = lossAccDict[f'Test Loss_{modelname}'] if name == 'Loss' else lossAccDict[f'Test Accuracy_{modelname}']
            axes[idx, i].plot(val_test, label='Testing')

            axes[idx, i].set_xlabel('Epoch')
            axes[idx, i].set_ylabel(name)
            axes[idx, i].legend()

#####################################################

def printTestAcc(testAccDict:dict,
                 modelnames:list,
                 **kwargs):
    """
    Print the last and max test accuracy of the model(s).

    Args:
        testAccDict (dict): Dictionary containing the test accuracy of the model(s).
        modelnames (list): List of model names.
    """
    if kwargs is not None:
        add_name = kwargs['add_name']
    # Get last and max test accuracy
    for modelname in modelnames:
        epochNum = testAccDict[f"Test Accuracy_{modelname}"].index(max(testAccDict[f"Test Accuracy_{modelname}"]))
        print(f'''{modelname}_{add_name} -> Last Test Accuracy: {testAccDict[f"Test Accuracy_{modelname}"][-1]} 
                Max Test Accuracy: {max(testAccDict[f"Test Accuracy_{modelname}"])} (Epoch:{epochNum})''')

#####################################################

def save_model(models:list,
               modelnames:list,
               epochDict:dict,
               path:str='models/'):
    """
    Save the model(s) and the loss and accuracy of the model(s) into files. More specifically, the model(s) will be saved
    as 'ConvNeXt_{modelname}.pt' and the loss and accuracy of the model(s) will be saved as 'ConvNeXt_{modelname}_EpochDict.json'.

    Args:
        models (list): List of models.
        modelnames (list): List of model names.
        epochDict (dict): Dictionary containing the loss and accuracy of the model(s).
        path (str, optional): Path to save the model(s) and the loss and accuracy of the model(s).
                    Default: 'models/'.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    for model, modelname in zip(models, modelnames):
        torch.save(model.state_dict(), f'{path}/{modelname}.pt')
        # Save the loss and accuracy for train and test set
        temp = dict()
        temp[f'Training Loss_{modelname}'] = epochDict[f'Training Loss_{modelname}']
        temp[f'Training Accuracy_{modelname}'] = epochDict[f'Training Accuracy_{modelname}']
        temp[f'Test Loss_{modelname}'] = epochDict[f'Test Loss_{modelname}']
        temp[f'Test Accuracy_{modelname}'] = epochDict[f'Test Accuracy_{modelname}']
        with open(f'{path}/{modelname}_EpochDict.json', 'w') as fp:
            json.dump(temp, fp)

#####################################################

def concatenateDict(*dicts:dict) -> dict:
    """
    Concatenate dictionaries.

    Args:
        *dicts (dict): Dictionaries to be concatenated.

    Returns:
        dict: Concatenated dictionary.
    """
    result = {}
    for dictionary in dicts:
        for key, value in dictionary.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value.copy()
    return result

#####################################################

def loadJSON(path:str) -> dict:
    """
    Load JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing the data from the JSON file.
    """
    with open(path) as f:
        data = json.load(f)
    return data

#####################################################
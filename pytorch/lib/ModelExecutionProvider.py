
import numpy as np
import torch
from torch import nn, optim

class ModelExecutionProvider:
    def __init__(self, model, optimizer=optim.Adam, lossfn=nn.CrossEntropyLoss, device="cuda:0"):
        """
            Initializes the ModelExecutionProvider. \n
            `model`: model to train \n
            `optimizer`: optimizer to use for training \n
            `lossfn`: loss function to use for training \n
            `device`: device to use for training \n
        """
        # Store the model
        self.model = model
        # Store the optimizer
        self.optimizer = optimizer(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        # Store the loss function
        self.lossfn = lossfn()
        # Store the device which should be used for training
        self.device = device
        # Move model to GPU instead of CPU
        self.model.to(self.device)

    def train_step(self, train_dl):
        """
            Performs a single training step on the model. \n
            `train_dl`: training data loader \n

            Returns the average loss and accuracy of the training step.
        """
        running_loss = 0.
        # Initialize variables to track training accuracy
        train_correct = 0
        train_total = 0
        # Enumerate over the data 
        amount_runs = 0
        for i, data in enumerate(train_dl):
            
            # Un-tuple the data
            inputs, labels = data

            # Move inputs and labels to the same device as the model
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients 
            self.optimizer.zero_grad()

            # Get Preds 
            outputs = self.model(inputs)

            # Get loss and gradients
            loss = self.lossfn(outputs, labels)
            loss.backward()

            # Run optimizer step
            self.optimizer.step()

            # Compute number of correct predictions
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Gather data and report
            running_loss += loss.item()
            amount_runs += 1
        avg_loss = running_loss / amount_runs
        train_acc = 100 * train_correct / train_total
        # print('epoch {} loss: {}'.format(epoch_index + 1, avg_loss))
        running_loss = 0.

        return avg_loss, train_acc

    def train(self, train_dl, test_dl, n_epochs=100):
        """
            Trains the model for `n_epochs` epochs. \n
            `train_dl`: training data loader \n
            `test_dl`: test data loader \n
            `n_epochs`: number of epochs to train the model for \n

            Returns the trained model with the best validation loss.
        """

        epoch_number = 0
        
        best_val_loss = np.inf

        # Variable to store the best model while training progresses. 
        best_model = None
        for epoch in range(n_epochs):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss, train_acc = self.train_step(train_dl=train_dl)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

           # Initialize variables to track validation accuracy
            val_correct = 0
            val_total = 0

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(test_dl):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(self.device)
                    vlabels = vlabels.to(self.device)

                    voutputs = self.model(vinputs)
                    vloss = self.lossfn(voutputs, vlabels)
                    running_vloss += vloss

                    _, predicted = torch.max(voutputs.data, 1)
                    val_total += vlabels.size(0)
                    val_correct += (predicted == vlabels).sum().item()

            avg_vloss = running_vloss / (i + 1)
            val_acc = 100 * val_correct / val_total
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print('ACCURACY train {} valid {}'.format(train_acc, val_acc))

            # Track best performance, and save the model's state
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
                best_model = self.model.state_dict()
            epoch_number += 1

        # Load the best model state and return it
        self.model.load_state_dict(best_model)
        return self.model

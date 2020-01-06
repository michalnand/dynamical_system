import dataset
import model

import torch

import dynamical_system


training_count  = 1000
testing_count   = 500

batch_size      = 32

ds  = dynamical_system.DynamicalSystem

dataset_training    = dataset.Create(ds, training_count, batch_size, device = "cuda")
dataset_testing     = dataset.Create(ds, testing_count, batch_size, device = "cuda")


input_shape  = dataset_training.get_input_shape()
output_shape = dataset_training.get_output_shape()

model = model.Create(input_shape[1], output_shape[1])

optimizer  = torch.optim.Adam(model.parameters(), lr= 0.001)


epoch_count = 100

for epoch in range(epoch_count):
    for i in range(training_count//batch_size):
        target_output, input = dataset_training.get_random_batch()

        predicted_output = model.forward(input)


        optimizer.zero_grad()

        loss = ((target_output - predicted_output)**2).mean() 
        loss.backward()
        optimizer.step()


    
    target_output, input = dataset_testing.get_random_batch()
    predicted_output = model.forward(input)

    loss = ((target_output - predicted_output)**2).mean() 

    print("EPOCH = ", epoch)
    for i in range(batch_size):
        print("target =    ", target_output[i].detach().to("cpu").numpy())
        print("predicted = ", predicted_output[i].detach().to("cpu").numpy())
        print("\n")

    print("LOSS = ", loss.detach().to("cpu").numpy())
    print("\n\n\n\n")
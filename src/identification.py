import oscilator
import dynamical_system

import torch


model_device = "cpu"

#target = oscilator.Oscilator(30.0, 1.9)

#create some random dynamical system
order = 4
target  = dynamical_system.DynamicalSystem(order, dt = 0.01, trainable=True, device=model_device)
target.set_random_transfer_matrix()

#create model, with same order
model  = dynamical_system.DynamicalSystem(order, dt = 0.01, trainable=True, device=model_device)
model.set_random_transfer_matrix()

#solver for model, use ADAM, works fine for nets
optimizer  = torch.optim.Adam(model.parameters(), lr= 0.1)

#proces identification, training
for step in range(10000):
    #set random state for target
    target.set_random_initial_state()
    
    #some initial conditions for model
    model.set_initial_state(target.get_y().detach().to("cpu").numpy())

    optimizer.zero_grad()

    #obtain trajectories, 128 points length
    target_trajectory = target.process(128)
    model_trajectory  = model.process(128)

    #RMS trajectory error
    loss = ((target_trajectory - model_trajectory)**2).mean()

    #error backpropagation
    loss.backward()

    #optimize
    optimizer.step()
    
    if step%10 == 0:
        print("target matrix = \n", target.get_a().detach().to("cpu").numpy())
        print("model matrix = \n", model.get_a().detach().to("cpu").numpy())
        print("loss = ", loss)
        print("\n")

#plot resulted trajectory
model.plot()

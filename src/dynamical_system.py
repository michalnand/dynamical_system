import torch
import numpy
import matplotlib.pyplot as plt

'''
general linear dynamical system
create, linear dynamical system :
dy = A*y

where 
    - A is transfer matrix
    - y is state vector
'''
class DynamicalSystem(torch.nn.Module):
    def __init__(self, y_order, dt = 0.01, trainable = False, device = "cpu"):
        super(DynamicalSystem, self).__init__()

        self.trainable  = trainable
        self.device     = device

        self.y_order    = y_order

        self.dt         = dt

        if self.trainable:
            self.a  = torch.nn.Parameter(torch.zeros(self.y_order, self.y_order), requires_grad = True).to(self.device)
        else:
            self.a  = torch.zeros((self.y_order, self.y_order)).to(device)

        self.y  = torch.zeros(self.y_order, requires_grad=self.trainable).to(device)



    def set_initial_state(self, initial_y):
        y = torch.from_numpy(initial_y).float().to(self.device)
        self.y = torch.tensor(y, requires_grad=self.trainable)


    def set_random_initial_state(self):
        self.y = torch.randn(self.y_order, requires_grad=self.trainable).to(self.device)


 
    def set_transfer_matrix(self, a):
        a_t = torch.from_numpy(a).float().to(self.device)

        if self.trainable:
            self.a  = torch.nn.Parameter(a_t, requires_grad = True).to(self.device)
        else:
            self.a  = a_t


    def set_random_transfer_matrix(self):
        if self.trainable:
            self.a  = torch.nn.Parameter(torch.randn(self.y_order, self.y_order), requires_grad = True).to(self.device)
        else:
            self.a  = torch.randn((self.y_order, self.y_order)).to(self.device)

    def get_y(self):
        return self.y

    def get_a(self):
        return self.a

    def step(self):
        self.dy = torch.matmul(self.y, self.a)*self.dt
        self.y  = self.y + self.dy
 
        
    def process(self, steps):
        self.result = torch.zeros((steps, self.y_order)).to(self.device)

        for n in range(steps):
            self.step()
            self.result[n] = self.get_y()

        return self.result

    def plot(self):
        result = self.result.detach().to("cpu").numpy()
        plt.plot(result)
        plt.show()

'''
ds = DynamicalSystem(2, 0.01, True, "cpu")

a = numpy.zeros((2, 2))

a[0][0] = -1.3
a[0][1] = -20.0
a[1][0] = 1.0

ds.set_transfer_matrix(a)



ds.set_random_initial_state()

print(ds.get_a().detach().to("cpu").numpy())
print(ds.get_y().detach().to("cpu").numpy())

ds.process(256)
ds.plot()
'''
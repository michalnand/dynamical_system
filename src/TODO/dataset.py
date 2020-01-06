import torch
import numpy


class Create:
    def __init__(self, dynamical_system, items_count = 10000, batch_size = 32, device = "cpu"):
        
        self.items_count = items_count
        self.seq_length = 256
        self.batch_size = batch_size

        self.order = 4

        self.dynamical_system = dynamical_system(self.order, 0.01)

        self.dynamical_system.set_random_transfer_matrix()

        self.dataset_x = torch.zeros(self.items_count, self.seq_length, self.order)
        self.dataset_y = torch.zeros(self.items_count, self.order*self.order)

        for i in range(self.items_count):
            y, x = self.create_item()

            self.dataset_x[i] = torch.from_numpy(x)
            self.dataset_y[i] = torch.from_numpy(y)

        self.input_shape    = (self.seq_length, self.order)
        self.output_shape   = (1, self.order*self.order)


        self.dataset_x.to(device)
        self.dataset_y.to(device)

        self.batch_x = torch.zeros(self.batch_size, self.seq_length, self.order).to(device)
        self.batch_y = torch.zeros(self.batch_size, self.order*self.order).to(device)



    
    def get_x(self):
        return self.dataset_x

    def get_y(self):
        return self.dataset_y

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.output_shape


    def get_random_batch(self):
        for i in range(self.batch_size):
            idx = numpy.random.randint(self.items_count)
            self.batch_x[i] = self.dataset_x[idx]
            self.batch_y[i] = self.dataset_y[idx]

        return self.batch_y, self.batch_x



    def create_item(self):
        self.dynamical_system.set_random_initial_state()
        
        x = self.dynamical_system.process(self.seq_length, 0.2)

        y = self.dynamical_system.get_a().copy()

        y = y.flatten()

        return y, x


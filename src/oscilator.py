import dynamical_system
import numpy


#special case of dynamical system, 2nd order harmonic oscilator, with dumping

class Oscilator(dynamical_system.DynamicalSystem):
    def __init__(self, frequency, dumping = 0.0, dt = 0.01):
        super().__init__(2, dt)
        
        a = numpy.zeros((2, 2))

        a[0][0] = -dumping
        a[0][1] = -frequency
        a[1][0] = 1.0
        a[1][1] = 0.0

        self.set_transfer_matrix(a)

        print(self.a)    

    def set_random(self):
        a = numpy.zeros((2, 2))

        a[0][0] = -numpy.random.rand()*2
        a[0][1] = -(numpy.random.rand())*30.0 + 2.0
        a[1][0] = 1.0
        a[1][1] = 0.0

        self.set_transfer_matrix(a)

  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class RosenbrockOpti():
    def __init__(self):
        #size of x is 2*dimension
        self.dimension=1
        self.x=np.zeros(2*self.dimension)
        self.iter_times=15000
        self.x1=[]
        self.x2=[]
        self.y=[]
        self.armjo()
        self.plot()

    def get_f(self,x):
        return np.sum(100*np.square(np.square(x[::2])- x[1::2]) + np.square(x[::2] - 1))
    def get_y(self,x1,x2):
        return 100*(x1**2-x2)**2+(x1-1)**2
        # return np.sum(100*np.square(np.square(x1)- x2) + np.square(x1 - 1))
    def cal_grad(self):
        grad=np.zeros_like(self.x)
        grad[::2]=400*self.x[::2]*(self.x[::2]**2-self.x[1::2])+2*(self.x[::2]-1)
        grad[1::2]=200*(self.x[1::2]-self.x[::2]**2)
        return grad
    
    def line_search(self):
        step=1
        while self.get_f(self.x)-self.get_f(self.x-step*self.cal_grad())<0.1*step*np.dot(self.cal_grad().T,self.cal_grad()):
            step=step/2
        return step
    
    def armjo(self):
        while self.iter_times>0:
            self.iter_times-=1
            self.x1.append(self.x[0])
            self.x2.append(self.x[1])
            self.y.append(self.get_f(self.x))
            step=self.line_search()
            self.x=self.x-self.cal_grad()*step
            if np.linalg.norm(self.x-np.ones_like(self.x))<1e-5:
                break
        print("answer = ")
        print(self.x)
        print("error")
        print(np.ones_like(self.x)-self.x)
        print("iteration times=")
        print(15000-self.iter_times)

    def plot(self):
        #plot3d
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(self.x1, self.x2, self.y,color='r',linewidth=6.0)
        x1=np.linspace(0,1,100)
        x2=np.linspace(0,1,100)
        x1,x2=np.meshgrid(x1,x2)
        y=self.get_y(x1,x2)
        # ax.plot3D(x1, x2, y)
        # ax.plot_surface(x1,x2,y,cmap=plt.get_cmap('rainbow'),alpha=0.6)
        ax.set_title('RosenbrockOpti')
        plt.show()
        
    def plot_(self):
        #plot3d
        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        ax=Axes3D(fig)
        x1=np.linspace(-1,1,100)
        x2=np.linspace(-1,1,100)
        print("size of x1 = ")
        print(x1.shape)
        x1,x2=np.meshgrid(x1,x2)
        print(x1.shape)
        y=self.get_y(x1,x2)
        print(y.shape)
        # ax.plot3D(x1, x2, y)
        ax.plot_surface(x1,x2,y,cmap=plt.get_cmap('rainbow'))
        ax.set_title('RosenbrockOpti')
        plt.show()

if __name__ == '__main__':
    optiRosenbrock = RosenbrockOpti()

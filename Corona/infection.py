import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def find_val(arr,val):
    return 

class Covid():

    def __init__(self, num_pts, num_cities, city_size, num_groc, groc_size, p_groc_hopp, groc_t_sigma, fam_size, fam_sigma, init_inf, r_inf, p_inf, inc_time, symp_time, sigma, dt):
        '''
        num_pts: 
        boxes:
        city_size:

        p_groc_hopp:
        groc_r_sigma:
        groc_t_sigma:
        
        fam_size:
        fam_sigma:

        init_inf:
        r_inf:
        p_inf:
        inc_time:
        symp_time:

        sigma:

        dt:
        '''
        self.num_pts = num_pts
        self.num_cities = num_cities
        self.city_size = city_size

        
        self.num_groc = num_groc
        self.groc_size = groc_size
        self.p_groc_hopp = p_groc_hopp
        self.groc_t_sigma = groc_t_sigma

        self.fam_size = fam_size
        self.fam_sigma = fam_sigma

        self.init_inf = init_inf
        self.r_inf = r_inf
        self.p_inf = p_inf
        self.inc_time = inc_time
        self.symp_time = symp_time

        self.sigma = sigma
        
        self.dt = dt
        self.t = [dt]

        self.init_pts()
        self.fig, self.axs = plt.subplots()
        
        self.fig.patch.set_facecolor((0.3,0.3,0.3))
        self.axs.set_facecolor((0.3,0.3,0.3))
        self.axs.set_xlim([0,self.city_size])
        self.axs.set_ylim([0,self.city_size])
        
        self.animation = FuncAnimation(self.fig, self.incr, interval=self.dt, init_func=self.init_plot)

    def init_pts(self):
        '''
        pts:    
        Array with size num_points and shape (x-coord, y-coord, colorcode, timestamp, groc, groc_t). This array describes the state of the system.
        
        fams: 
        Array with family sizes and mean grid position. This will be used to construct the state by populating each family's members
        with the mean position and some uniform position noise.
        '''
        self.pts = np.zeros((self.num_pts,5))
        
        self.fams = {'sizes': np.random.randint(1,self.fam_size), 'mus': np.random.randint(1,self.fam_size,2)}
        self.pts[:self.fams['sizes'],0] = (np.random.rand(self.fams['sizes'])-0.5)*self.fam_sigma + self.fams['mus'][0]
        self.pts[:self.fams['sizes'],1] = (np.random.rand(self.fams['sizes'])-0.5)*self.fam_sigma + self.fams['mus'][1]
        
        while np.sum(self.fams['sizes']) + self.fam_size <= self.num_pts:
            
            _fam_mu = np.random.randint(1,self.city_size,2)
            try:
                if _fam_mu[0] not in self.fams['mus'][0] and _fam_mu[1] not in self.fams['mus'][1]:
                    self.fams['mus'] = np.vstack( (self.fams['mus'], _fam_mu))
                    self.fams['sizes'] = np.vstack( (np.random.randint(1,self.fam_size), self.fams['sizes']) )
            except TypeError as _:
                self.fams['mus'] = np.vstack( (self.fams['mus'], _fam_mu))
                self.fams['sizes'] = np.vstack( (np.random.randint(1,self.fam_size), self.fams['sizes']) )
            
            self.pts[np.sum(self.fams['sizes'][:-1]):np.sum(self.fams['sizes']),0] = (np.random.rand(self.fams['sizes'][-1,0])-0.5)*self.fam_sigma + self.fams['mus'][-1,0]
            self.pts[np.sum(self.fams['sizes'][:-1]):np.sum(self.fams['sizes']),1] = (np.random.rand(self.fams['sizes'][-1,0])-0.5)*self.fam_sigma + self.fams['mus'][-1,1]
        
        self.pts[:self.init_inf,2] = 3
        self.pts[self.init_inf:,2] = 0

    def init_plot(self):
        self.scat = self.axs.scatter(self.pts[:,0], self.pts[:,1], c=self.pts[:,2], s=2, cmap='brg')
        return self.scat,

    def incr(self, timestamp):
        self.infect(timestamp)
        self.symptoms(timestamp)
        self.immune(timestamp)

        self.groc_hopp(timestamp)
        self.move()

        self.scat.set_offsets(self.pts[:,:2])
        self.scat.set_array(self.pts[:,2])

        self.t.append(sum(self.t)+self.dt)
        return self.scat, 

    def infect(self, timestamp):
        '''
        Calculate point to point distance in order to check wether its smaller than the infection radius r_inf.
        If this is true and one of the points is in state 2 or 3 infection will happen with probabililty p_inf 
        and the corresponding timestamp will be noted.
        '''
        for i in range(self.num_pts):
            for j in range(i+1,self.num_pts):

                d = np.sqrt((self.pts[j,0]-self.pts[i,0])**2+(self.pts[j,1]-self.pts[i,1])**2)
                
                if( ((self.pts[i,2] == 0 and self.pts[j,2] in [2,3]) or (self.pts[j,2] == 0 and self.pts[i,2] in [2,3])) and d<self.r_inf):
                    if np.random.rand() < self.p_inf:
                        if self.pts[i,2] == 0:
                            self.pts[i,2] = 3
                            self.pts[i,3] = timestamp
                        else:
                            self.pts[j,2] = 3
                            self.pts[j,3] = timestamp
        
        return self.pts

    def symptoms(self, timestamp):
        for i in range(self.num_pts):
            if self.pts[i,2] == 3 and (timestamp - self.pts[i,3] > self.inc_time):
                self.pts[i,2] = 2
                self.pts[i,3] = timestamp

    def immune(self, timestamp):
        for i in range(self.num_pts):
            if self.pts[i,2] == 2 and (timestamp - self.pts[i,3] > self.symp_time):
                self.pts[i,2] = 1
                self.pts[i,3] = timestamp
    
    def groc_hopp(self, timestamp):
        for i in range(self.fams['sizes'].size):

            _fam_groc = np.random.randint(self.fams['sizes'][i,0])
            if 0 in self.pts[np.sum(self.fams['sizes'][:i]):np.sum(self.fams['sizes'][:i+1]),4]:

                if np.random.rand() < self.p_groc_hopp and self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,2] != 2: 
                    self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,4] = 1
                    self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,0] = self.fams['mus'][i,0] + (self.city_size/2 - self.fams['mus'][i,0])*(1-np.abs(1-self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,4]/50))
                    self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,1] = self.fams['mus'][i,1] + (self.city_size/2 - self.fams['mus'][i,1])*(1-np.abs(1-self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,4]/50))
                
            else:                    
                _fam_groc = np.argmin(np.abs(self.pts[np.sum(self.fams['sizes'][:i]):np.sum(self.fams['sizes'][:i+1]),4]-1))
                if self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,0] != self.fams['mus'][i,0]:
                    self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,0] = self.fams['mus'][i,0] + (self.city_size/2 - self.fams['mus'][i,0])*(1-np.abs(1-self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,4]/50))
                    self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,1] = self.fams['mus'][i,1] + (self.city_size/2 - self.fams['mus'][i,1])*(1-np.abs(1-self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,4]/50))
                    self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,4] += 1
                else:
                    self.pts[np.sum(self.fams['sizes'][:i])+_fam_groc,4] = 0

    def fam_hopp(self):
        pass

    def move(self):
        for i in range(self.fams['sizes'].size):

            self.pts[np.sum(self.fams['sizes'][:i]):np.sum(self.fams['sizes'][:i+1]),0] += (np.random.rand(self.fams['sizes'][i,0])-0.5)*self.fam_sigma
            self.pts[np.sum(self.fams['sizes'][:i]):np.sum(self.fams['sizes'][:i+1]),1] += (np.random.rand(self.fams['sizes'][i,0])-0.5)*self.fam_sigma

    def rand_walk(self, i):
        for i in range(self.num_pts):
            
            for j in range(i+1,self.num_pts):
                d = np.sqrt((self.pts[j,0]-self.pts[i,0])**2+(self.pts[j,1]-self.pts[i,1])**2)
                if d<self.r_inf:
                    if np.random.rand() < self.p_inf:
                        self.pts[i,2] = self.pts[i,2] or self.pts[j,2]
                        self.pts[j,2] = self.pts[i,2]
            
            if self.pts[i,0]>self.sigma/2 and self.city_size-self.pts[i,0]>self.sigma/2:
                self.pts[i,0] += self.sigma*(np.random.rand()-0.5)
            elif self.pts[i,0]<self.sigma/2:
                self.pts[i,0] += self.sigma*np.random.rand()*0.5
            else:
                self.pts[i,0] += self.sigma*(np.random.rand()-1)*0.5

            if self.pts[i,1]>self.sigma/2 and self.city_size-self.pts[i,1]>self.sigma/2:
                self.pts[i,1] += self.sigma*(np.random.rand()-0.5)
            elif self.pts[i,1]<self.sigma/2:
                self.pts[i,1] += self.sigma*np.random.rand()*0.5
            else:
                self.pts[i,1] += self.sigma*(np.random.rand()-1)*0.5

        self.scat.set_offsets(self.pts[:,:2])
        self.scat.set_array(self.pts[:,2])
        return self.scat,

cov = Covid(num_pts=10, num_cities=2, city_size=100, 
            num_groc=1, groc_size=10, p_groc_hopp=0.05, groc_t_sigma=5, 
            fam_size=3, fam_sigma=0.2, 
            init_inf=1, r_inf=0.1, p_inf=0.15, inc_time = 1000, symp_time = 2000, sigma=2, dt=200)
plt.show()
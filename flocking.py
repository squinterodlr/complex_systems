#-*- encoding:utf-8 -*-

#Libraries

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

#Constants
NUMBIRDS = 100
NEIGHDIST = 0.1
WALLTHRES = 0.01
STAY = 0.01
COHESE = 0.08
ALIGN = 0.08
AVOID = 0.005
AVOIDWALL = 0.0
MAXRAND = 0.01
MAXSPD = 2.0
MINSPD = 0.1
TORUS = 1
dt = 0.01

#################
#Routines:
#################

#Normalization of a vector
def normalize(vect):

    norm = np.linalg.norm(vect)
    if norm < np.finfo(float).eps:
        return vect
    else:
        return vect/norm

#Distance as measured in a torus (AKA periodic boundary conditions)
def torusdist(p1,p2):

    x1,y1,z1 = p1
    x2,y2,z2 = p2
    dx1 = dx2 = dy1 = dy2 = dz1 = dz2 = dx = dy = dz = 0

    if x2 > x1:
        dx1 = x2 - x1
        dx2 = x1 - x1 - 1
    else:
        dx1 = x2 - x1
        dx2 = x2 - x1 + 1
    if np.abs(dx1) < np.abs(dx2):
        dx = dx1
    else:
        dx = dx2

    if y2 > y1:
        dy1 = y2 - y1
        dy2 = y1 - y1 - 1
    else:
        dy1 = y2 - y1
        dy2 = y2 - y1 + 1
    if np.abs(dy1) < np.abs(dy2):
        dy = dy1
    else:
        dy = dy2
    
    if z2 > z1:
        dz1 = z2 - z1
        dz2 = z1 - z1 - 1
    else:
        dz1 = z2 - z1
        dz2 = z2 - z1 + 1
    if np.abs(dz1) < np.abs(dz2):
        dz = dz1
    else:
        dz = dz2
    
    return np.sqrt(dx*dx + dy*dy + dz*dz)

#Routine to measure distance, either on torus or R3
def distance(p1,p2):
    if TORUS == 0:
        return np.linalg.norm(p2-p1)
    else:
        return torusdist(p1,p2)

# Integration routine
def timestep(boid_list):
    
    for boid in boid_list:
        boid.move(boid_list)

#Write from list of boids to list of positions and velocities
def write_to_list(boid_list):

    n = len(boid_list)

    pos_list = np.zeros([NUMBIRDS,3])
    vel_list = np.zeros([NUMBIRDS,3])

    for i in xrange(n):

        boid = boid_list[i]
        pos_list[i] = boid.pos
        vel_list[i] = boid.vel

    return pos_list,vel_list

#####################
#CLASSES
#####################

#Define the Boid class
class Boid(object):
    
    def __init__(self,ID,pos,vel):
        self.ID = ID
        self.pos = pos
        self.vel = vel

    #Time integration
    def move(self,boid_list):

        #simple Euler
        newpos = self.pos + self.vel*dt
        self.pos = newpos

        #If outside of box, move inside as torus
        self.move_inside()

        #Update velocity according to flocking rules
        self.flock(boid_list)

    def flock(self,boid_list):
        
        #Find all neighbors and flocking velocities
        self.get_neigh(boid_list)
        v_cohese = self.cohese(boid_list)
        v_align = self.align(boid_list)
        v_avoid = self.avoid_other(boid_list)
        v_rand = -MAXRAND + 2*MAXRAND*np.random.rand(3)
        v_walls = self.avoid_walls()
        
        #Add new flocking velocity
        flock_vel = (1.0 + STAY)*self.vel + v_cohese + v_align + v_avoid + v_rand + v_walls
        flock_spd = np.linalg.norm(flock_vel)

        #If new speed is greater than max, make it max
        if flock_spd > MAXSPD:
            flock_vel = MAXSPD*normalize(flock_vel)
        
        elif flock_spd < MINSPD:
            flock_vel = MINSPD*normalize(flock_vel)

        self.vel = flock_vel
        
    #Find the neighborhood of each boid
    def get_neigh(self,boid_list):
        
        #initialize neighborhood
        self.neigh = []
        
        for boid in boid_list:
            
            #Do not count self in neighborhood
            if boid == self:
                continue

            dist = distance(self.pos,boid.pos)

            if dist <= NEIGHDIST:
                self.neigh.append(boid.ID)
        
    #get closer to center of mass of nhood
    def cohese(self,boid_list):
        
        mean_pos = np.array([0,0,0])
        n = 0
        
        #find c.o.m of neighborhood
        for i in self.neigh:
            friend = boid_list[i]
            mean_pos = mean_pos + friend.pos
            n += 1
        
        #if neighborhood is empty, do nothing
        #align to center of mass of friends
        if n != 0:
            mean_pos = mean_pos/n
            v_cohese = COHESE*(mean_pos - self.pos)

        else:
            v_cohese = np.array([0,0,0])

        return v_cohese

    #Boid aligns to velocity of friends
    def align(self,boid_list):

        mean_vel = np.array([0,0,0])
        n = 0
        
        #find mean velocity of friends
        for i in self.neigh:
            friend = boid_list[i]
            mean_vel = mean_vel + np.array(friend.vel)
            n += 1
        
        #if neighborhood is empty, do nothing
        #align to mean velocity of friends
        if n != 0:
            mean_vel = mean_vel/n
            v_align = ALIGN*(mean_vel - self.vel)
        else:
            v_align = np.array([0,0,0])
            
        return v_align

    #Boids try not to collide with other boids
    def avoid_other(self,boid_list):

        avoid_vect = np.array([0,0,0])
        n = 0
        
        #If boid has no friends, do nothing
        if len(self.neigh) == 0:
            return avoid_vect

        #Move away from friends, each weighed by inverse distance
        else:
            for i in self.neigh:
                n = n+1
                friend = boid_list[i]
                dist = distance(self.pos, friend.pos)
                avoid_vect = avoid_vect + AVOID*(np.array(self.pos) - np.array(friend.pos))/dist
                
            return avoid_vect/n

    #Routine to move inside bounding box in periodic BCs
    def move_inside(self):
        
        x,y,z = self.pos

        if x >= 1:
            x = x - 1
        elif x < 0:
            x = x + 1

        if y >= 1:
            y = y - 1
        elif y < 0:
            y = y + 1

        if z >= 1:
            z = z - 1
        elif z < 0:
            z = z + 1
        
        self.pos = np.array([x,y,z])

    #Rules to avoid walls
    def avoid_walls(self):

        x,y,z = self.pos
        vx = vy = vz = 0

        #Avoid harder the closer to wall
        if x < WALLTHRES:
            vx = AVOIDWALL/x
        elif 1 - x < WALLTHRES:
            vx = AVOIDWALL/(x-1)


        if y < WALLTHRES:
            vy = AVOIDWALL/y
        elif 1 - y < WALLTHRES:
            vy = AVOIDWALL/(y-1)

        if z < WALLTHRES:
            vz = AVOIDWALL/z
        elif 1 - z < WALLTHRES:
            vz = AVOIDWALL/(z-1)

        return np.array([vx,vy,vz])


#################
#Initialize#
#################



#initial positions
Rs = np.random.rand(NUMBIRDS,3)

#initial velocities
Vs = -0.5*MAXSPD + MAXSPD*np.random.rand(NUMBIRDS,3)
mean_v = np.mean(Vs,axis=0)
meanspd = np.linalg.norm(mean_v)
spddisp = np.std(np.linalg.norm(Vs,axis=1))

#create boid list
birds = []

for i in xrange(NUMBIRDS):
    birds.append(Boid(i,Rs[i],Vs[i]))
#########
#ANIMATE
########

plt.cla()


#Create axes, dots for boids and text functions
Nframes = 1500
fig = plt.figure(figsize = (8,8))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
line,  = ax.plot(Rs[:,0],Rs[:,1],Rs[:,2],'k.')

speed_text = "mean speed = %f" %(meanspd)
spdtext = ax.text2D(0.1,0.9,speed_text, transform=ax.transAxes)

stdtext = "std = %f" %(spddisp)
disptext = ax.text2D(0.1,0.85, stdtext, transform=ax.transAxes)

ax.set_aspect('equal')

def init():
    line.set_data([],[])
    line.set_3d_properties([])
    spdtext.set_text("")
    disptext.set_text("")
    return line, spdtext, disptext

def animate(i):

    if (i+1)%10 == 0:
        print "Frame %d of %d" %(i+1,Nframes)

    global Rs, Vs, meanspd, speed_text, spddisp, stdtext

    #draw position and update text

    line.set_data(Rs[:,0],Rs[:,1])
    line.set_3d_properties(Rs[:,2])
    
    meanspd = np.linalg.norm(np.mean(Vs,axis=0))
    speed_text = "mean speed = %f" %(meanspd)
    spdtext.set_text(speed_text)

    spddisp = np.std(np.linalg.norm(Vs,axis=1))
    stdtext = "std = %f" %(spddisp)
    disptext.set_text(stdtext)

    fig.canvas.draw()
    
    #Update position of birds
    timestep(birds)
    Rs, Vs = write_to_list(birds)

    return line, spdtext, disptext

anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True,interval=10,frames=Nframes)

#Save animation
anim.save('flocking.mp4', fps=30, writer="ffmpeg", codec="libx264")
plt.cla()

#Rejoice
print ("Done!")

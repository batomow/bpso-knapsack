from mylib.bpso import Bparticle as BP, Sack, fitness as fx, Item
from numpy import array as nparray
from numpy import random as rng
from numpy import clip as npclip
from numpy import zeros as npzeros
from numpy import average as npaverage
from numpy import std as npstd
from math import e
import matplotlib.pyplot as plt

#=========================================================================#
#                            Definitions .                                #
#=========================================================================#

SACK_SIZE = 10
SACK_CAPACITY = 165
EXPERIMENTS = 10
SWARM_SIZE = 10
STEPS = 10
ALPHA:float = 1
BETA:float = 0.75
VMAX:float = 1
INERTIA:float = 0.8

sack = Sack([
    Item(23, 92), 
    Item(31, 57),
    Item(29, 49),
    Item(44, 68),
    Item(53, 60),
    Item(38, 43), 
    Item(63, 67),
    Item(85, 84),
    Item(89, 87),
    Item(82, 72)],
    SACK_CAPACITY)

def find_best(): 
    best_score = 0
    best_particle:BP = swarm[0]
    for particle in swarm: 
        score = fx(sack, particle)
        if score > best_score: 
            best_particle = particle
            best_score = score
    return best_particle

def update_velocity(particle:BP, global_best:BP, local_best:BP):
    E1 = nparray([rng.random() for n in range(SACK_SIZE)]) # [0.1, 0.2, 0.002, 0.4, ...]
    E2 = nparray([rng.random() for n in range(SACK_SIZE)])
    v1 = global_best.solution - particle.solution
    v2 = local_best.solution - particle.solution
    velocity = ALPHA * E1 * v1 + BETA * E2 * v2
    velocity = npclip(particle.velocity, -VMAX, VMAX)
    particle.velocity = particle.velocity * INERTIA + velocity

def sigmoid(value): 
    return 1 / ( 1 + (e**(-value)))

def update_position(particle:BP):
    for i in range(SACK_SIZE): 
        r = rng.random()
        p = sigmoid(particle.velocity[i])
        if r < p: 
            particle.solution[i] = 1
        else: 
            particle.solution[i] = 0

#==========================================================================#
#                               Algorithm .                                #
#==========================================================================#

global_best:BP
local_best:BP

# init
swarm = [BP(SACK_SIZE) for s in range(SWARM_SIZE)]
global_best = find_best()
local_best = swarm[rng.randint(0, SWARM_SIZE)]
best_score = fx(sack, local_best)
global_best_score = fx(sack, global_best)
results = npzeros((STEPS, EXPERIMENTS)) #matriz de resultados

for iteracion in range(EXPERIMENTS): # numero de experimentos
    for epoch in range(STEPS):  # cuantos pasos van a dar las particulas
        # optimize
        for p in swarm: 
            if p is global_best: 
                continue
            if p is local_best: 
                continue
            update_velocity(p, global_best, local_best)
            update_position(p)
            score = fx(sack, p)
            if score > best_score: 
                best_score = score
                local_best = p
            if score > global_best_score: 
                global_best_score = score
                global_best = p
        results[epoch, iteracion] = best_score
    print('[%d] best_solution: ' % iteracion, global_best, fx(sack, global_best))

averages = npaverage(results, 1).reshape(STEPS, 1)
deviations = npstd(results, 1).reshape(STEPS, 1)
dMinusAverages = averages - deviations # promedio menos la desviacion
dPlusAverages = averages + deviations # promedio mas la desviacion 

#graficar los datos
plt.plot(range(STEPS), dMinusAverages)
plt.plot(range(STEPS), averages)
plt.plot(range(STEPS), dPlusAverages)
plt.show() 

print('solution')
total_weight = 0
total_money = 0
for s in range(SACK_SIZE):
    if global_best.solution[s]: 
        print(sack.items[s])
        total_weight += sack.items[s].weight
        total_money += sack.items[s].price
print('total weight: ', total_weight, 'total money: ', total_money)
from numpy import array as nparray
from numpy import random as rng

#item definition for knapsack with weight and price
class Item:
    def __init__(self, weight:float, price:float): 
        self.weight = weight
        self.price = price
    
    @property
    def value(self): 
        return self.price/self.weight
    
    def __str__(self): 
        return str('<%i w, %i $, %.4f $/w>') % (self.weight, self.price, self.value)

#binary particle
class Bparticle: 
    def __init__(self, size:int):
        self.solution:nparray = rng.randint(2, size=size)
        self.velocity:nparray = nparray([0 for n in range(size)], dtype=float)
    
    def __str__(self): 
        result = [str(e) for e in self.solution]
        return '[' + ', '.join(result) + ']'

class Sack: 
    def __init__(self, items:[Item], capacity:float): 
        self.items:[Item] = items
        self.capacity = capacity
    
    def __str__(self): 
        result = 'knapsack\n'
        for i in self.items: 
            result += str(i) + '\n'
        return result
    
    @property
    def total_weigth(self): 
        total:float = 0
        for i in self.items: 
            total += i.weight
        return str("%.4f w") % total
    
    @property
    def total_value(self): 
        total:float = 0
        for i in self.items: 
            total += i.value
        return str("%.4f $") % total

#fitness function calculation for knapsack
def fitness(sack:Sack, particle):
    val:float = 0
    total_weight:float = 0
    for n in range(len(sack.items)): 
        val += particle.solution[n] * sack.items[n].value
        total_weight += particle.solution[n] * sack.items[n].weight
        if total_weight > sack.capacity: 
            return 0
        
    return val

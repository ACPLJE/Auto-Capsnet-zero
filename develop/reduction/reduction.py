import numpy as np
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from eedge import edge
from nnode import node
from torchvision.datasets import MNIST
from timeit import timeit
from rredu import redu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_channels = out_channels
       

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: out_capsules * (batch_size, out_channels, height, weight)
        
        out = self.conv(x)
        # Flatten out: (batch_size, out_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        #return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)
        return squash(out.contiguous().view(batch_size, -1, self.out_channels))


class DigitCaps(nn.Module):
    """Digit capsule layer."""

    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing):
        """
        Initialize the layer.
        Args:
            in_dim: 		Dimensionality of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            out_caps: 		Number of capsules in the capsule layer
            out_dim: 		Dimensionality, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True)
       

    def forward(self, x):
        
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        # W @ x =
        # (1, out_caps, in_caps, out_dim, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, out_caps, in_caps, out_dims, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, out_caps, in_caps, out_dim)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, out_caps, in_caps, 1) -> Softmax along out_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, out_caps, in_caps, 1) * (batch_size, in_caps, out_caps, out_dim) ->
            # (batch_size, out_caps, in_caps, out_dim) sum across in_caps ->
            # (batch_size, out_caps, out_dim)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along out_dim
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, out_caps, in_caps, out_dim) @ (batch_size, out_caps, out_dim, 1)
            # -> (batch_size, out_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along out_dim
        
        v = squash(s)
        
        return v


class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""

    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv2d layer
        self.conv = nn.Conv2d(1, 256, 9)
        self.relu = nn.ReLU(inplace=True)

       
        
        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=32,
                                        in_channels=256,
                                        out_channels=8,
                                        kernel_size=9,
                                        stride=2)

        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=8,
                                    in_caps=32 * 6 * 6,
                                    out_caps=10,
                                    out_dim=16,
                                    num_routing=3)

        # Reconstruction layer
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid())

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.primary_caps(out)
        out = self.digit_caps(out)

        # Shape of logits: (batch_size, out_capsules)
        logits = torch.norm(out, dim=-1)
        pred = torch.eye(10).to(device).index_select(dim=0, index=torch.argmax(logits, dim=1))

        # Reconstruction
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * pred.unsqueeze(2)).contiguous().view(batch_size, -1))

        return logits, reconstruction


class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-4
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, logits, reconstructions):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)

        # Reconstruction loss
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)

        # Combine two losses
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss


#new squash function using evolutionary search algorithm is consisted of primitive math operations.

def Redu():
    return ['norm','sum','mean', 'max','min','std','var']


def Edge():
    return ['add','sub','mul','div']


def Node():
    
    return ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh','exp', 'exp1', 'exp2','expm1','relu', 'sigmoid', 'tanh', 'softplus', 'softsign', 'elu', 'selu', 'celu', 'gelu', 'hardshrink', 'hardtanh', 'leakyrelu', 'logsigmoid', 'rrelu', 'sqrt', 'square', 'cube', 'log', 'log2', 'log10'] #'norm','sum'

population_size = 10
nodenum = 2
edgenum = nodenum - 1
#edgenum = np.random.randint(1,((nodenum * (nodenum-1)*1/2)+1))
relasize = edgenum+1

def generate_population():
    """ generate population of squash functions. """
    
    population = []
    for i in range(population_size):
        
        npopulation = (np.random.choice(Redu(), size = 1),np.random.choice(Node(), size = nodenum-1))
        epopulation = np.random.choice(Edge(), size = edgenum)
        #rela = np.random.randint(nodenum, size = relasize)
        rela = random.sample(range(0,nodenum),relasize) 
        population.append([npopulation,epopulation, rela])
        #population.append([npopulation,epopulation])
    return population
     

    
    #cpu cost that prifiling cpu time of new squash function
def gpu_cost():
    """ return the cpu cost of population. """
       
    return gpu_time
        
    

    #define fitness function
def fitness_function(population):
    """ return the fitness of population. """
    fitness = []
    global current_pop
   
    for i in range(len(population)):
        current_pop = population[i]
        #fitness.append(model_accuracy(population))
        fitness.append([model_accuracy(), gpu_cost()])
       
    return fitness


def squash(x):
    """ return the squash function of x. """
    global gpu_time 
    
    nsq=[]
    rpop = current_pop[0][0]
    npop = current_pop[0][1] 
    
    epop = current_pop[1]
    rela = current_pop[2]
    #print(npop, rpop, epop, rela)
    #print(rpop)
    #print(epop)
    #print(rela)
    squash = 0
    sq = 0
    
 
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    #start = time.process_time()
    for i in range(len(rpop)):
        
        s = redu(x,rpop[i])
        nsq.append(s) 
    for i in range(len(npop)):
                
        s = node(x,npop[i])
        nsq.append(s)  
        
     
    starter.record()
    for i in range(len(rela)-1):
        if i == 0:
            nsq1 = nsq[rela[i]]
            nsq2 = nsq[rela[i+1]]
            
        if i != 0:
            nsq1 = sq
            nsq2 = nsq[rela[i+1]]
        
        sq = edge(nsq1,nsq2,epop[i])
        
  #  for i in range(edgenum):
  #      if i == 0:
  #          nsq1 = nsq[i]
  #          nsq2 = nsq[i+1]
  #          sq = edge(nsq1,nsq2,epop[i])
            
  #      if i != 0:
  #          nsq1 = sq
  #          nsq2 = nsq[i+1]
        
   #         sq = edge(nsq1,nsq2,epop[i])
        
    squash = sq
    ender.record()
    
    torch.cuda.synchronize()
    gpu_time = starter.elapsed_time(ender)
    #end = time.process_time()
    #cpu_time = end - start 
    
    if squash.dim() == 0:
        squash = x*100
    else:
        squash = squash
    return squash

def fitness_scaling(fitness):
    fitness = np.array(fitness)
    accufit = fitness[:, 0]
    gpufit = fitness[:, 1] 
    fit1 = accufit/ accufit.sum()
    fit1 = 9 * fit1
    fit2 = 1 / gpufit
    fit2 = fit2/ fit2.sum()
    fit2 = 1 * fit2
    fitness = fit1 + fit2
    fitness = fitness/fitness.sum()
    return fitness

    #evaluate model accuracy with new squash function
def model_accuracy():
    """ return the model accuracy of population. """
    
    model = CapsNet().to(device)
    criterion = CapsuleLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    # Load data
    transform = transforms.Compose([
        # shift by 2 pixels in either direction with zero padding.
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    DATA_PATH = './data'
    BATCH_SIZE = 128
    train_loader = DataLoader(
        dataset=MNIST(root=DATA_PATH, download=True, train=True, transform=transform),
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True)
    test_loader = DataLoader(
        dataset=MNIST(root=DATA_PATH, download=True, train=False, transform=transform),
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True)

    # Train
    EPOCHES = 1
    model.train()
    for ep in range(EPOCHES):
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
            logits, reconstruction = model(images)

            # Compute loss & accuracy
            loss = criterion(images, labels, logits, reconstruction)
            correct += torch.sum(
                torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss
            loss.backward()
            optimizer.step()
            #print('Epoch {}, loss: {}, accuracy: {}'.format(ep + 1,
            #                                                total_loss / batch_id,
            #                                                accuracy))
            batch_id += 1
        scheduler.step(ep)
       # print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))

    # Eval
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        # Add channels = 1
        images = images.to(device)
        # Categogrical encoding
        labels = torch.eye(10).index_select(dim=0, index=labels).to(device) 
        logits, reconstructions = model(images)
        pred_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(pred_labels == torch.argmax(labels, dim=1)).item()
        total += len(labels)
    #Pint('Accuracy: {}'.format(correct / total))
    accuracy = correct / total
    return accuracy
   
    
global bestsquash

class searching_new_squash_function(nn.Module):
    def __init__(self):
        super(searching_new_squash_function, self).__init__()
        #self.population = generate_population()
        
        #self.fitness = fitness_function(self.population)     
        self.generation = 0
        self.generation_size = 50
        self.population_size = 5
        self.parent_size = 1
        self.mutation_rate = 0.2
        self.mutation_size = 1
        

    def new_generation(self):
        """ return new generation of population. """
        
        self.generation += 1
        self.population = generate_population()
        if self.generation == 1:
            self.fitness = fitness_function(self.population)
            self.population = self.selection()
            #self.population = self.crossover()
            self.population = self.mutation()
            self.fitness = fitness_function(self.population)
            self.bestsquash = self.best_squash()
            
            return self.population
        
        elif self.generation > 1:
            self.fitness = fitness_function(self.population)
            self.population[0] = self.bestsquash
            self.fitness = fitness_function(self.population)
            self.population = self.selection()
            #self.population = self.crossover()
            self.population = self.mutation()
            self.fitness = fitness_function(self.population)
            self.bestsquash = self.best_squash()
            return self.population
   
        
    
    def selection(self):
        """ return selected population. """  
            
        population = []
        population_index = 0
        self.population_index = np.arange(population_size)
     
        fitness= fitness_scaling(self.fitness)
        
        for i in range(self.population_size):
            population_index = np.random.choice(self.population_index, p=fitness)
            population.append(self.population[population_index])
        #print(population)
        return population
       
  
      
        

    #def crossover(self):
    #    """ return crossovered population. """
    #    population = []
    #    fitness = np.array(self.fitness)
    #    accufit = fitness[:, 0]
    #    cpufit = fitness[:, 1] 
    #    #fit1 = 9 * accufit/ accufit.sum()
    #    #fit2 = 1 * cpufit/ cpufit.sum()
    #    fit1 = accufit/ accufit.sum()
    #    fit1 = 2 * fit1
    #    fit2 = 1 / cpufit
    #    fit2 = fit2/ fit2.sum()
    #    fit2 = 8 * fit2
    #    fitness = fit1 + fit2
    #    fitness = fitness/fitness.sum()
    #    population_index = 0
    #    parindex = 0
    #    #print(fitness)
    #    self.population_index = np.arange(self.population_size)
    #    for i in range(self.population_size):
    #        parent = np.random.choice(self.population_index, self.population_size)
    #        child = []
    #        for i in range(self.parent_size):
    #            #child.append(np.random.choice(parent)[:,i])
    #            parindex = np.random.choice(parent, p=fitness)
    #            child.append(self.population[parindex])
    #        population.extend(child)
    #   
    #    return population

    def mutation(self):
        """ return mutationed population. """
        population = []
        mutated = []
        f = -1
        for i in self.population:
            f += 1
            for _ in range(self.mutation_size - 1):
                if np.random.rand() < self.mutation_rate:
                    
                    npopulation = (np.random.choice(Redu(), size = 1),np.random.choice(Node(), size = nodenum-1))
                    epopulation = np.random.choice(Edge(), size = edgenum)
                    rela = random.randint(nodenum, size = edgenum-1)
                    #i = [npopulation,epopulation]
                    i = [npopulation,epopulation, rela]
                  
                      
                else:
                    i = self.population[f]
            population.append(i)   
            
        return population
    
    def best_squash(self):
        fitness= fitness_scaling(self.fitness)
        bestsquash = self.population[np.argmax(fitness)]
        return bestsquash

    def search(self):
        """ return the best population. """
        for i in range(self.generation_size):
            self.new_generation()
            
        fitness= fitness_scaling(self.fitness)
        
        print('self.population[np.argmax(fitness): {}'.format(self.population[np.argmax(fitness)]))
        return self.population[np.argmax(fitness)]


def main():

    start_time = time.time()
    print("Start")
    new_squash_function = searching_new_squash_function()
    new_squash_function.search()
    print("End")
    print("Time: {:.4f}sec".format((time.time() - start_time)))
    print(new_squash_function.search())
    
if __name__ == '__main__':
    main()
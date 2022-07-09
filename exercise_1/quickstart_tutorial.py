
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


batch_size = 64
max_epochs = 5

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]





# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
        
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)





def train(dataloader, model, loss_fn, optimizer):
    for t in range(max_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        for phase in ['train', 'val']:
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            test_loss, correct = 0, 0
            
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
              
              
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Compute prediction error
                    pred = model(X)
                    loss = loss_fn(pred, y)
                    
                    if phase == 'train':
                       
                        # Backpropagation
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if batch % 100 == 0:
                            loss, current = loss.item(), batch * len(X)
                            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                        
                    elif phase == 'val':
                        test_loss += loss.item()
                        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                        
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        
    print("Done!")
    # save model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")


def test(dataloader, model):
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')





if __name__ == '__main__':

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    # creating models
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    #optimizing model parameters and saving the trained models
    train(dataloader, model, loss_fn, optimizer)
    
    
    #test
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    
    test(dataloader, model)

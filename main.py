import torch
from models import SurvivalNet

x = torch.randn(2, 5, 50, 50, 50).cuda()
age = torch.randn(2, 1, 512).cuda() # must be 512
y = torch.randn(2, 5).cuda()

model = SurvivalNet(brain_input_channels=5)

# criterion = torch.nn.CrossEntropyLoss(reduction='sum')
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(1500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x, age)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 10 == 9:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
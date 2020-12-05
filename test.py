import torch
from models import Unet3D

# from data_sampler import (get_train_dataflow, get_eval_dataflow, get_test_dataflow)
# from eval import (eval_brats, pred_brats, segment_one_image, segment_one_image_dynamic)


x = torch.randn(2, 2, 38, 38, 38)
y = torch.randn(2, 128, 1, 1, 1)

model = Unet3D(2)

try:
    print(torch.cuda.current_device())
    torch.cuda.set_device(0)
    print(torch.cuda.current_device())
    x = x.cuda()
    y = y.cuda()
    model.cuda()
except:
    pass

# criterion = torch.nn.CrossEntropyLoss(reduction='sum')
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(1500):

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
    # print(y_pred.size())

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 10 == 9:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

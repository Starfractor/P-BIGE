import torch
import torch.nn as nn
from dataset import dataset_MOT_segmented_surrogate
from tqdm import tqdm
from write_mot import write_muscle_activations

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        
        # Define MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten the input (assuming x is of shape [batch_size, 196, 33])
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to [batch_size, 196*33]
        
        # Pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output is [batch_size, 196*80]
        
        # Reshape output to [batch_size, 196, 80]
        x = x.view(batch_size, 196, 80)
        
        return x

train_loader = dataset_MOT_segmented_surrogate.DATALoader("mcs",
                                        32,
                                        window_size=60,
                                        unit_length=4)

test_loader = dataset_MOT_segmented_surrogate.DATALoader("mcs",
                                        32,
                                        window_size=60,
                                        unit_length=4,
                                        mode='test')

# train_loader_iter = dataset_MOT_MCS.cycle(train_loader)
# train_loader_iter = dataset_MOT_segmented.cycle(train_loader)

# Hyperparameters
input_dim = 196 * 33  # Flattened input dimension
hidden_dim = 1024      # Example hidden dimension
output_dim = 196 * 80 # Flattened output dimension
num_epochs = 10

# Instantiate the model
model = MLPModel(input_dim, hidden_dim, output_dim)

# Example input (batch_size = 32, 196 time steps, 33 features)
input_tensor = torch.randn(32, 196, 33)
output_tensor = model(input_tensor)

print(output_tensor.shape)  # Should print torch.Size([32, 196, 78])

# Loss function and optimizer
criterion = nn.MSELoss()  # Example for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop (with test loss calculation)
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0  # Track training loss for the epoch
    
    for inputs, _, targets, _ in train_loader:
        optimizer.zero_grad()  # Clear gradients
        
        # Forward pass
        inputs = inputs.float()
        outputs = model(inputs)
        
        # Compute loss
        targets = targets.float()
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()  # Accumulate loss for reporting

    # Calculate average training loss
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

    # Evaluation phase on test data
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    test_l = 0.0
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, _, targets, _ in test_loader:
            inputs = inputs.float()
            outputs = model(inputs)
            
            # Compute test loss
            targets = targets.float()
            loss = criterion(outputs, targets)
            
            l = criterion(outputs[:,:,:-4], targets[:,:,:-4])
            test_l += l.item()
            
            test_loss += loss.item()  # Accumulate test loss for averaging

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    print(f"Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}")
    avg_test_l = test_l / len(test_loader)
    print(f"Epoch {epoch+1}, Test Loss: {avg_test_l:.4f}")

final_test_loader = dataset_MOT_segmented_surrogate.DATALoader("mcs",
                                        1,
                                        window_size=60,
                                        unit_length=4,
                                        mode='test')

for inputs,_,_,name in final_test_loader:
    
    inputs = inputs.float()
    outputs = model(inputs)
    name = name[0]
    session_id = name.split('/')[6]
    folder = name.split('/')[9]
    trial = folder.split('_')[0]
    segment = folder.split('_')[2]
    act_name = session_id + "_" + trial + "_" + segment
    
    outputs = outputs.detach().cpu().numpy()
    write_muscle_activations("/home/ubuntu/data/MCS_DATA/surrogate_output/"+act_name+".mot", outputs[0])
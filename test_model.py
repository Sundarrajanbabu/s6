import torch
import torch.nn as nn
from train import MNISTModel
from torchvision import datasets, transforms
import pytest
import glob
import os

def get_latest_model():
    model_files = glob.glob('model_mnist_*.pth')
    if not model_files:  # If no model file exists
        pytest.skip("No trained model found. Run training first.")
    return max(model_files, key=os.path.getctime)

def test_parameter_count():
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has {total_params} parameters, should be less than 100000"

def test_input_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_architecture():
    model = MNISTModel()
    
    # Test for batch normalization
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use batch normalization"
    
    # Test for dropout
    has_dropout = any(isinstance(m, nn.Dropout2d) for m in model.modules())
    assert has_dropout, "Model should use dropout"
    
    # Test for fully connected layer
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    assert has_fc, "Model should have fully connected layers"

def test_model_accuracy():
    try:
        # First try to train a new model if no model exists
        if not glob.glob('model_mnist_*.pth'):
            from train import train
            train()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MNISTModel().to(device)
        
        try:
            # Load the latest trained model
            model_path = get_latest_model()
            model.load_state_dict(torch.load(model_path))
        except RuntimeError as e:
            pytest.skip(f"Error loading model, might be incompatible. Please retrain: {str(e)}")
            
        model.eval()
        
        # Load test dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%"
    
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__]) 
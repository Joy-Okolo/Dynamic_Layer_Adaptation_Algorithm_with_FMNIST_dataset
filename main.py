import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
import torchvision
import torchvision.transforms as transforms
import time
import random
import copy
import numpy as np
import os
import sys
import json
from collections import defaultdict

# Set up plotting backend for cluster environment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# Cluster environment setup
def setup_cluster_environment():
    """Setup environment for cluster execution"""

    # Set matplotlib backend
    plt.ioff()  # Turn off interactive plotting

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ CUDA not available, using CPU")

    # Create necessary directories
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    print("🔧 Cluster environment setup complete")

# Modify the plotting functions to save instead of show
def save_plot_instead_of_show():
    """Replace plt.show() with plt.savefig() for cluster"""
    original_show = plt.show

    def cluster_show(filename_prefix="plot"):
        timestamp = int(time.time())
        filename = f"./plots/{filename_prefix}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Plot saved: {filename}")
        plt.close()  # Close to free memory

    plt.show = cluster_show

# Memory management for large experiments
def optimize_for_cluster():
    """Optimize settings for cluster environment"""

    # Reduce batch size if memory constrained
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 8:  # Less than 8GB
            print("⚠️  Limited GPU memory detected, reducing batch size")
            return {'batch_size': 32, 'num_clients': 25}

    return {'batch_size': 64, 'num_clients': 50}

# Add error handling wrapper
def safe_experiment_runner(func, *args, **kwargs):
    """Run experiment with error handling"""
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ GPU out of memory! Trying with reduced settings...")
            torch.cuda.empty_cache()
            # Reduce parameters and try again
            if 'num_clients' in kwargs:
                kwargs['num_clients'] = max(10, kwargs['num_clients'] // 2)
            if 'max_local_epochs' in kwargs:
                kwargs['max_local_epochs'] = max(5, kwargs['max_local_epochs'] // 2)
            return func(*args, **kwargs)
        else:
            raise e
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 1. Enhanced Fashion-MNIST Neural Network ---
class FashionMNISTNet(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128, 64], num_classes=10, dropout_rate=0.25):
        super(FashionMNISTNet, self).__init__()
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:  # No dropout before last hidden layer
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- 2. Apply layer mask to control backprop ---
def apply_layer_mask(model, layers_to_update):
    """
    Apply mask to update only the last 'layers_to_update' layers.
    """
    all_layers = [module for module in model.model.modules()
                  if isinstance(module, nn.Linear)]

    total_linear_layers = len(all_layers)
    layers_to_freeze = max(0, total_linear_layers - layers_to_update)

    for i, layer in enumerate(all_layers):
        requires_grad = (i >= layers_to_freeze)
        if hasattr(layer, 'weight'):
            layer.weight.requires_grad = requires_grad
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.requires_grad = requires_grad

# --- 3. Enhanced Local Training with Early Stopping ---
def train_locally_with_early_stopping(model, train_loader, val_loader, layers_to_update, device,
                                     client_speed=1.0, max_epochs=15, lr=0.001, patience=5):
    """
    Enhanced local training with early stopping and validation monitoring for Fashion-MNIST
    """
    model = copy.deepcopy(model).to(device)
    apply_layer_mask(model, layers_to_update)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    start_time = time.time()

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []

    model.train()
    for epoch in range(max_epochs):
        # Training phase
        epoch_train_loss = 0
        num_batches = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            num_batches += 1

            # Simulate different client speeds
            if client_speed < 1.0:
                time.sleep(0.001 * (1.0 - client_speed))

        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        model.train()  # Back to training mode

        # Stop early if patience exceeded
        if patience_counter >= patience:
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    elapsed_time = (time.time() - start_time) / client_speed
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_accuracy = val_accuracies[-1] if val_accuracies else 0
    epochs_trained = len(train_losses)

    return (model.state_dict(), elapsed_time, final_train_loss,
            final_val_accuracy, epochs_trained, train_losses, val_losses, val_accuracies)

# --- 4. Improved Layer depth adaptation for Fashion-MNIST ---
def adjust_layers_improved(elapsed_time, current_depth, client_id,
                         T_low=10.0, T_high=30.0, max_layers=5):
    """
    Improved layer adjustment with thresholds optimized for Fashion-MNIST
    """
    # Client-specific thresholds accounting for Fashion-MNIST complexity
    client_T_low = T_low * (0.7 + client_id * 0.1)
    client_T_high = T_high * (0.8 + client_id * 0.2)

    if elapsed_time > client_T_high and current_depth > 1:
        return current_depth - 1
    elif elapsed_time < client_T_low and current_depth < max_layers:
        return current_depth + 1
    else:
        return current_depth

# --- 4b. Enhanced Stability tracking for DLA ---
class DLAStabilityTracker:
    """
    Enhanced stability tracking with Fashion-MNIST specific adaptations
    """
    def __init__(self, num_clients):
        self.adaptation_history = {cid: [] for cid in range(num_clients)}
        self.consecutive_suggestions = {cid: {'increase': 0, 'decrease': 0, 'stay': 0}
                                      for cid in range(num_clients)}
        self.performance_history = {cid: [] for cid in range(num_clients)}

    def should_adapt(self, client_id, current_depth, suggested_depth, client_performance=None, stability_threshold=3):
        """
        Enhanced adaptation decision with performance consideration
        """
        # Record performance if provided
        if client_performance is not None:
            self.performance_history[client_id].append(client_performance)

        if suggested_depth == current_depth:
            self.consecutive_suggestions[client_id] = {'increase': 0, 'decrease': 0, 'stay': 0}
            return current_depth

        # Track suggestion type
        if suggested_depth > current_depth:
            self.consecutive_suggestions[client_id]['increase'] += 1
            self.consecutive_suggestions[client_id]['decrease'] = 0
            self.consecutive_suggestions[client_id]['stay'] = 0

            # Only increase if we've had multiple consecutive suggestions
            if self.consecutive_suggestions[client_id]['increase'] >= stability_threshold:
                self.consecutive_suggestions[client_id]['increase'] = 0
                return suggested_depth

        elif suggested_depth < current_depth:
            self.consecutive_suggestions[client_id]['decrease'] += 1
            self.consecutive_suggestions[client_id]['increase'] = 0
            self.consecutive_suggestions[client_id]['stay'] = 0

            # Only decrease if we've had multiple consecutive suggestions
            if self.consecutive_suggestions[client_id]['decrease'] >= stability_threshold:
                self.consecutive_suggestions[client_id]['decrease'] = 0
                return suggested_depth

        return current_depth

# --- 5. Enhanced Model evaluation ---
def evaluate_model(model, test_loader, device):
    """
    Enhanced model evaluation with detailed metrics
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss

# --- 6. Weighted Average of Model Weights ---
def average_weights(weight_list, client_sizes=None):
    """
    Weighted average of model weights based on client dataset sizes
    """
    if client_sizes is None:
        client_sizes = [1] * len(weight_list)

    total_size = sum(client_sizes)
    avg_weights = copy.deepcopy(weight_list[0])

    # Initialize with zeros
    for key in avg_weights:
        avg_weights[key] = torch.zeros_like(avg_weights[key])

    # Weighted sum
    for i, weights in enumerate(weight_list):
        weight = client_sizes[i] / total_size
        for key in avg_weights:
            avg_weights[key] += weights[key] * weight

    return avg_weights

# --- 7. Enhanced Fashion-MNIST federated dataset creation ---
def create_federated_fashion_mnist(num_clients=50, batch_size=64, iid=True, val_split=0.2):
    """
    Create enhanced federated Fashion-MNIST dataset with validation splits
    """
    print("Downloading Fashion-MNIST dataset...")

    # Fashion-MNIST specific preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST specific normalization
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Fashion-MNIST class names
    fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Enhanced client creation with full dataset utilization
    total_samples = len(train_dataset)
    base_samples_per_client = total_samples // num_clients

    clients = []
    client_speeds = []

    print(f"Creating federated Fashion-MNIST data splits for {num_clients} clients...")

    for client_id in range(num_clients):
        # More realistic speed distribution for Fashion-MNIST
        if client_id < num_clients * 0.1:  # 10% slow clients (mobile devices)
            speed_factor = np.random.uniform(0.3, 0.6)
        elif client_id < num_clients * 0.2:  # 10% fast clients (edge servers)
            speed_factor = np.random.uniform(1.8, 2.5)
        else:  # 80% normal clients (laptops/desktops)
            speed_factor = np.random.uniform(0.8, 1.4)

        # Fixed sample allocation for consistency
        client_samples = base_samples_per_client

        if iid:
            # IID: random samples from all fashion categories
            indices = np.random.choice(range(total_samples), client_samples, replace=False)
        else:
            # Non-IID: fashion stores with specialized inventories
            num_categories_per_client = np.random.randint(3, 7)  # 3-6 fashion categories per client
            target_classes = np.random.choice(range(10), size=num_categories_per_client, replace=False)
            indices = []
            targets = np.array(train_dataset.targets)

            for target_class in target_classes:
                class_indices = np.where(targets == target_class)[0]
                class_samples = min(len(class_indices), client_samples // len(target_classes))
                selected = np.random.choice(class_indices, class_samples, replace=False)
                indices.extend(selected)

            indices = indices[:client_samples]

            if client_id < 5:  # Show first 5 clients for debugging
                client_fashion_types = [fashion_classes[i] for i in target_classes]
                print(f"Client {client_id} fashion categories: {client_fashion_types}")

        # Create train/validation split
        client_dataset = Subset(train_dataset, indices)
        train_size = int((1 - val_split) * len(client_dataset))
        val_size = len(client_dataset) - train_size

        train_subset, val_subset = random_split(client_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        clients.append((train_loader, val_loader, len(client_dataset)))
        client_speeds.append(speed_factor)

        if client_id < 10 or client_id % 10 == 0:  # Progress indicator
            print(f"Client {client_id}: {len(client_dataset)} samples, speed {speed_factor:.2f}x")

    print(f"✅ Created {num_clients} Fashion-MNIST clients with {total_samples} total samples")
    return clients, test_loader, client_speeds

# --- 8. Global Early Stopping Tracker ---
class GlobalEarlyStopping:
    """
    Track global model performance and implement early stopping for Fashion-MNIST
    """
    def __init__(self, patience=18, min_delta=0.1):  # Slightly higher patience for Fashion-MNIST
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = 0
        self.patience_counter = 0
        self.should_stop = False
        self.best_round = 0

    def update(self, current_accuracy, current_round):
        """
        Update early stopping status based on current accuracy
        """
        if current_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = current_accuracy
            self.patience_counter = 0
            self.best_round = current_round
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.should_stop = True

        return self.should_stop

# --- 9. Enhanced Fashion-MNIST Federated Training Loop ---
def federated_training_loop_fashion_mnist(
    clients,
    client_speeds,
    max_rounds,
    device,
    strategy="fedavg",
    max_layers=5,
    fixed_depth=3,
    test_loader=None,
    lr=0.001,
    max_local_epochs=15,
    early_stop_patience=18
):
    """
    Enhanced federated training specifically optimized for Fashion-MNIST
    """
    global_model = FashionMNISTNet().to(device)

    # Enhanced metrics tracking
    results = {
        'test_accuracies': [],
        'test_losses': [],
        'training_times': defaultdict(list),
        'client_depths': defaultdict(list),
        'client_epochs_trained': defaultdict(list),
        'client_val_accuracies': defaultdict(list),
        'round_times': [],
        'losses': defaultdict(list),
        'convergence_info': {}
    }

    # Initialize components
    current_depths = {cid: 3 for cid in range(len(clients))}  # Start at depth 3 for Fashion-MNIST
    dla_tracker = DLAStabilityTracker(len(clients)) if strategy == "dla" else None
    global_early_stopping = GlobalEarlyStopping(patience=early_stop_patience)

    print(f"\n🚀 Starting Enhanced Fashion-MNIST Federated Training")
    print(f"🔄 Strategy: {strategy.upper()}")
    print(f"👥 Clients: {len(clients)}")
    print(f"🔄 Max Rounds: {max_rounds}")
    print(f"🏋️ Max Layers: {max_layers}")
    print(f"📚 Max Local Epochs: {max_local_epochs}")
    print(f"⏰ Early Stop Patience: {early_stop_patience}")
    print(f"💻 Device: {device}")
    print(f"🏃 Client speeds: min={min(client_speeds):.2f}x, max={max(client_speeds):.2f}x, avg={np.mean(client_speeds):.2f}x")
    print("-" * 80)

    for r in range(max_rounds):
        round_start_time = time.time()
        print(f"\n--- Round {r+1}/{max_rounds} ---")

        local_weights = []
        local_sizes = []
        round_stats = {
            'total_epochs': 0,
            'early_stops': 0,
            'avg_val_accuracy': 0,
            'depth_changes': 0
        }

        for cid, (train_loader, val_loader, dataset_size) in enumerate(clients):
            # Determine layer depth based on strategy
            if strategy == "fedavg":
                layer_depth = max_layers
            elif strategy == "fedpmt":
                layer_depth = fixed_depth
            elif strategy == "feddrop":
                layer_depth = random.randint(1, max_layers)
            elif strategy == "dla":
                layer_depth = current_depths[cid]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Record depth
            results['client_depths'][cid].append(layer_depth)

            # Enhanced local training with early stopping
            training_results = train_locally_with_early_stopping(
                global_model, train_loader, val_loader, layer_depth, device,
                client_speed=client_speeds[cid], max_epochs=max_local_epochs, lr=lr, patience=5
            )

            (local_state_dict, elapsed_time, final_train_loss,
             final_val_accuracy, epochs_trained, train_losses, val_losses, val_accuracies) = training_results

            # Record comprehensive metrics
            results['training_times'][cid].append(elapsed_time)
            results['losses'][cid].append(final_train_loss)
            results['client_epochs_trained'][cid].append(epochs_trained)
            results['client_val_accuracies'][cid].append(final_val_accuracy)

            # Update round statistics
            round_stats['total_epochs'] += epochs_trained
            round_stats['avg_val_accuracy'] += final_val_accuracy
            if epochs_trained < max_local_epochs:
                round_stats['early_stops'] += 1

            # Show progress for first few clients and every 10th client
            if cid < 5 or cid % 10 == 0:
                print(f"Client {cid:2d}: depth={layer_depth}, epochs={epochs_trained:2d}/{max_local_epochs}, "
                      f"time={elapsed_time:5.1f}s, val_acc={final_val_accuracy:5.1f}%, loss={final_train_loss:.4f}")

            # Update depth for DLA strategy
            if strategy == "dla":
                suggested_depth = adjust_layers_improved(elapsed_time, layer_depth, cid,
                                                       T_low=10.0, T_high=30.0, max_layers=max_layers)

                new_depth = dla_tracker.should_adapt(cid, layer_depth, suggested_depth,
                                                    client_performance=final_val_accuracy, stability_threshold=3)
                current_depths[cid] = new_depth

                if new_depth != layer_depth:
                    if cid < 5:  # Only show first 5 for clarity
                        print(f"  → Client {cid} depth adapted: {layer_depth} → {new_depth}")
                    round_stats['depth_changes'] += 1

            # Collect weights and sizes for aggregation
            local_weights.append(local_state_dict)
            local_sizes.append(dataset_size)

        # Aggregate weights
        global_weights = average_weights(local_weights, local_sizes)
        global_model.load_state_dict(global_weights)

        # Evaluate global model
        if test_loader:
            test_acc, test_loss = evaluate_model(global_model, test_loader, device)
            results['test_accuracies'].append(test_acc)
            results['test_losses'].append(test_loss)

            # Update round statistics
            round_stats['avg_val_accuracy'] /= len(clients)

            print(f"\n📊 Round {r+1} Summary:")
            print(f"   Global Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
            print(f"   Avg Client Val Acc: {round_stats['avg_val_accuracy']:.2f}%")
            print(f"   Total Epochs Trained: {round_stats['total_epochs']}")
            print(f"   Early Stops: {round_stats['early_stops']}/{len(clients)}")
            if strategy == "dla":
                print(f"   Depth Changes: {round_stats['depth_changes']}")

        round_time = time.time() - round_start_time
        results['round_times'].append(round_time)
        print(f"   Round Time: {round_time:.1f}s")

        # Check global early stopping
        if test_loader and global_early_stopping.update(test_acc, r):
            print(f"\n🛑 Global Early Stopping triggered at round {r+1}")
            print(f"   Best accuracy: {global_early_stopping.best_accuracy:.2f}% at round {global_early_stopping.best_round+1}")
            results['convergence_info'] = {
                'stopped_early': True,
                'stopped_at_round': r + 1,
                'best_accuracy': global_early_stopping.best_accuracy,
                'best_round': global_early_stopping.best_round + 1
            }
            break
    else:
        # Training completed without early stopping
        results['convergence_info'] = {
            'stopped_early': False,
            'completed_rounds': max_rounds,
            'final_accuracy': results['test_accuracies'][-1] if results['test_accuracies'] else 0
        }

    return global_model, results

# --- 10. Enhanced Fashion-MNIST Plotting Functions ---
def plot_fashion_mnist_comprehensive_results(all_results, strategies):
    """
    Comprehensive visualization specifically for Fashion-MNIST results
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    # 1. Test Accuracy Evolution
    axes[0, 0].set_title('Fashion-MNIST Test Accuracy Over Rounds', fontsize=14, fontweight='bold')
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies']:
            rounds = range(1, len(results['test_accuracies'])+1)
            axes[0, 0].plot(rounds, results['test_accuracies'],
                           marker=markers[i], label=strategy.upper(), color=colors[i],
                           linestyle=linestyles[i], linewidth=2.5, markersize=6,
                           markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[i])

    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_facecolor('#fafafa')

    # 2. Training Time Comparison
    axes[0, 1].set_title('Average Training Time per Round', fontsize=14, fontweight='bold')
    avg_times = []
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['round_times']:
            avg_time = np.mean(results['round_times'])
            avg_times.append(avg_time)
            bar = axes[0, 1].bar(strategy.upper(), avg_time, color=colors[i], alpha=0.8, edgecolor='black')
            axes[0, 1].text(i, avg_time + max(avg_times)*0.01, f'{avg_time:.1f}s',
                           ha='center', va='bottom', fontweight='bold')

    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].set_facecolor('#fafafa')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Convergence Analysis
    axes[0, 2].set_title('Convergence Information', fontsize=14, fontweight='bold')
    convergence_data = []
    for strategy, results in zip(strategies, all_results):
        conv_info = results.get('convergence_info', {})
        if conv_info.get('stopped_early', False):
            convergence_data.append(f"{strategy.upper()}: Early stop at round {conv_info['stopped_at_round']}")
        else:
            convergence_data.append(f"{strategy.upper()}: Completed {conv_info.get('completed_rounds', 'N/A')} rounds")

    axes[0, 2].text(0.1, 0.9, '\n'.join(convergence_data), transform=axes[0, 2].transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis('off')

    # 4. Layer Depth Evolution (DLA)
    axes[1, 0].set_title('Layer Depth Evolution (DLA)', fontsize=14, fontweight='bold')
    dla_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    for strategy, results in zip(strategies, all_results):
        if strategy == "dla" and results['client_depths']:
            # Show first 5 clients for clarity
            for cid in range(min(5, len(results['client_depths']))):
                depths = results['client_depths'][cid]
                if depths:
                    axes[1, 0].plot(range(1, len(depths)+1), depths,
                                   marker='s', label=f'Client {cid}', color=dla_colors[cid],
                                   linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=2)

    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Layer Depth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_facecolor('#fafafa')

    # 5. Training Efficiency
    axes[1, 1].set_title('Training Efficiency Analysis', fontsize=14, fontweight='bold')
    efficiency_data = []
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies'] and results['round_times']:
            final_accuracy = results['test_accuracies'][-1]
            total_time = sum(results['round_times'])
            efficiency = final_accuracy / (total_time / 60)  # Accuracy per minute
            efficiency_data.append(efficiency)

            bar = axes[1, 1].bar(strategy.upper(), efficiency, color=colors[i], alpha=0.8, edgecolor='black')
            axes[1, 1].text(i, efficiency + max(efficiency_data)*0.01, f'{efficiency:.1f}',
                           ha='center', va='bottom', fontweight='bold')

    axes[1, 1].set_ylabel('Accuracy per Minute')
    axes[1, 1].set_facecolor('#fafafa')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 6. Final Performance Summary
    axes[1, 2].set_title('Final Performance Summary', fontsize=14, fontweight='bold')
    summary_text = []
    for strategy, results in zip(strategies, all_results):
        if results['test_accuracies']:
            final_acc = results['test_accuracies'][-1]
            best_acc = max(results['test_accuracies'])
            total_time = sum(results['round_times']) if results['round_times'] else 0

            # Calculate average epochs per client
            avg_epochs = 0
            if results['client_epochs_trained']:
                all_epochs = [epoch for client_epochs in results['client_epochs_trained'].values()
                             for epoch in client_epochs]
                avg_epochs = np.mean(all_epochs) if all_epochs else 0

            summary_text.append(
                f"{strategy.upper()}:\n"
                f"  Final: {final_acc:.1f}%\n"
                f"  Best: {best_acc:.1f}%\n"
                f"  Time: {total_time:.0f}s\n"
                f"  Avg Epochs: {avg_epochs:.1f}\n"
            )

    axes[1, 2].text(0.05, 0.95, '\n'.join(summary_text), transform=axes[1, 2].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show("comprehensive_results")

def plot_fashion_mnist_accuracy_only(all_results, strategies):
    """
    High-visibility accuracy plot specifically for Fashion-MNIST
    """
    plt.figure(figsize=(14, 8))

    colors = ['#000080', '#FF8C00', '#228B22', '#DC143C']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies']:
            rounds = range(1, len(results['test_accuracies'])+1)
            plt.plot(rounds, results['test_accuracies'],
                    marker=markers[i], label=f'{strategy.upper()}', color=colors[i],
                    linestyle=linestyles[i], linewidth=4, markersize=8,
                    markerfacecolor='white', markeredgewidth=3, markeredgecolor=colors[i])

    plt.title('Fashion-MNIST Test Accuracy: Enhanced Federated Learning\n(50 Clients, 15 Epochs, Research-Grade)',
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, framealpha=0.95, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.gca().set_facecolor('#f8f9fa')

    # Add annotations for final points
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies']:
            final_acc = results['test_accuracies'][-1]
            plt.annotate(f'{final_acc:.1f}%',
                        xy=(len(results['test_accuracies']), final_acc),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7),
                        color='white')

    plt.tight_layout()
    plt.show("accuracy_only")

# --- 11. Main Fashion-MNIST Execution Function ---
def run_fashion_mnist_federated_experiment(iid=True, num_rounds=100,
                                         num_clients=50, max_local_epochs=15, early_stop_patience=18):
    """
    Run enhanced Fashion-MNIST federated learning experiment with comprehensive monitoring
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # Create enhanced federated Fashion-MNIST dataset
    data_type = "IID" if iid else "Non-IID"
    print(f"👗 Creating federated Fashion-MNIST dataset ({data_type})...")
    clients, test_loader, client_speeds = create_federated_fashion_mnist(
        num_clients=num_clients,
        batch_size=64,
        iid=iid,
        val_split=0.2
    )

    strategies = ["fedavg", "fedpmt", "feddrop", "dla"]
    all_results = []

    print(f"\n{'='*80}")
    print(f"👗 ENHANCED FASHION-MNIST FEDERATED LEARNING EXPERIMENT")
    print(f"📊 Dataset: Fashion-MNIST ({data_type})")
    print(f"👥 Clients: {num_clients}")
    print(f"🔄 Max Rounds: {num_rounds}")
    print(f"📚 Max Local Epochs: {max_local_epochs}")
    print(f"⏰ Early Stop Patience: {early_stop_patience}")
    print(f"🎨 Fashion Categories: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot")
    print(f"{'='*80}")

    # Run experiments for each strategy
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"🔄 Running {strategy.upper()} Strategy on Fashion-MNIST")
        print(f"{'='*80}")

        set_seed(42)  # Reset seed for fair comparison

        start_time = time.time()
        global_model, results = federated_training_loop_fashion_mnist(
            clients=clients,
            client_speeds=client_speeds,
            max_rounds=num_rounds,
            device=device,
            strategy=strategy,
            max_layers=5,
            fixed_depth=3,
            test_loader=test_loader,
            lr=0.001,
            max_local_epochs=max_local_epochs,
            early_stop_patience=early_stop_patience
        )

        experiment_time = time.time() - start_time
        all_results.append(results)

        # Print comprehensive summary
        if results['test_accuracies']:
            final_acc = results['test_accuracies'][-1]
            best_acc = max(results['test_accuracies'])
            total_training_time = sum(results['round_times'])

            # Calculate average client metrics
            avg_epochs_per_round = 0
            total_early_stops = 0
            if results['client_epochs_trained']:
                all_epochs = []
                for round_epochs in zip(*results['client_epochs_trained'].values()):
                    all_epochs.extend(round_epochs)
                avg_epochs_per_round = np.mean(all_epochs) if all_epochs else 0
                total_early_stops = sum(1 for epochs in all_epochs if epochs < max_local_epochs)

            conv_info = results.get('convergence_info', {})

            print(f"\n📊 {strategy.upper()} FINAL RESULTS:")
            print(f"   🎯 Final Test Accuracy: {final_acc:.2f}%")
            print(f"   🏆 Best Test Accuracy: {best_acc:.2f}%")
            print(f"   ⏱️  Total Training Time: {total_training_time:.1f}s")
            print(f"   🕒 Total Experiment Time: {experiment_time:.1f}s")
            print(f"   📚 Avg Epochs per Client: {avg_epochs_per_round:.1f}")
            print(f"   ⏹️  Total Early Stops: {total_early_stops}")

            if conv_info.get('stopped_early', False):
                print(f"   🛑 Early stopping at round: {conv_info['stopped_at_round']}")
                print(f"   🎯 Best accuracy: {conv_info['best_accuracy']:.2f}% at round {conv_info['best_round']}")
            else:
                print(f"   ✅ Completed all {conv_info.get('completed_rounds', num_rounds)} rounds")

    # Generate comprehensive visualizations
    print(f"\n{'='*80}")
    print("📊 GENERATING FASHION-MNIST ANALYSIS PLOTS")
    print(f"{'='*80}")

    plot_fashion_mnist_comprehensive_results(all_results, strategies)
    plot_fashion_mnist_accuracy_only(all_results, strategies)

    # Print final comparison table
    print(f"\n{'='*80}")
    print(f"📋 FINAL COMPARISON SUMMARY - FASHION-MNIST")
    print(f"{'='*80}")
    print(f"{'Strategy':<10} | {'Final':<6} | {'Best':<6} | {'Time':<7} | {'Efficiency':<10} | {'Status':<15}")
    print("-" * 80)

    for i, strategy in enumerate(strategies):
        if all_results[i]['test_accuracies']:
            final_acc = all_results[i]['test_accuracies'][-1]
            best_acc = max(all_results[i]['test_accuracies'])
            total_time = sum(all_results[i]['round_times'])
            efficiency = final_acc / (total_time / 60)  # Accuracy per minute

            conv_info = all_results[i].get('convergence_info', {})
            status = f"Early@{conv_info['stopped_at_round']}" if conv_info.get('stopped_early') else "Completed"

            print(f"{strategy.upper():<10} | {final_acc:5.1f}% | {best_acc:5.1f}% | {total_time:6.0f}s | {efficiency:8.1f} | {status:<15}")

    # Print DLA-specific analysis
    dla_results = None
    fedavg_results = None
    for i, strategy in enumerate(strategies):
        if strategy == "dla":
            dla_results = all_results[i]
        elif strategy == "fedavg":
            fedavg_results = all_results[i]

    if dla_results and fedavg_results and dla_results['test_accuracies'] and fedavg_results['test_accuracies']:
        dla_final = dla_results['test_accuracies'][-1]
        fedavg_final = fedavg_results['test_accuracies'][-1]
        dla_time = sum(dla_results['round_times'])
        fedavg_time = sum(fedavg_results['round_times'])

        accuracy_ratio = (dla_final / fedavg_final) * 100
        time_savings = ((fedavg_time - dla_time) / fedavg_time) * 100

        print(f"\n{'='*80}")
        print(f"🎯 DLA PERFORMANCE ANALYSIS - FASHION-MNIST")
        print(f"{'='*80}")
        print(f"📊 DLA achieved {accuracy_ratio:.1f}% of FedAvg's accuracy")
        print(f"⏰ DLA provided {time_savings:.1f}% time savings vs FedAvg")
        print(f"🎨 Fashion-MNIST complexity: Perfect testbed for partial training evaluation")
        print(f"👗 Realistic scenario: Fashion retailers with specialized inventories")

    return all_results

# --- 12. Main Execution with Cluster Optimizations ---
if __name__ == "__main__":
    # Setup cluster environment
    setup_cluster_environment()
    save_plot_instead_of_show()

    # Get optimized settings for cluster
    cluster_settings = optimize_for_cluster()

    # Research-grade configuration for Fashion-MNIST
    NUM_CLIENTS = cluster_settings['num_clients']
    MAX_LOCAL_EPOCHS = 15
    NUM_ROUNDS = 100
    EARLY_STOP_PATIENCE = 18  # Slightly higher for Fashion-MNIST complexity

    print("👗 Starting Enhanced Fashion-MNIST Federated Learning Research Experiment")
    print(f"⚙️  Configuration: {NUM_CLIENTS} clients, {MAX_LOCAL_EPOCHS} epochs, {NUM_ROUNDS} rounds")
    print(f"🎨 Fashion-MNIST: 10 clothing categories, 60K training samples, 10K test samples")

    # Run Fashion-MNIST experiment with error handling
    print("\n" + "="*100)
    print("👗 RUNNING ENHANCED FASHION-MNIST EXPERIMENT")
    print("="*100)

    fashion_mnist_results = safe_experiment_runner(
        run_fashion_mnist_federated_experiment,
        iid=True,
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
        max_local_epochs=MAX_LOCAL_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE
    )

    # Uncomment to run Non-IID experiment (Fashion stores with specialized inventories)
    """
    print("\n" + "="*100)
    print("🏪 RUNNING NON-IID FASHION-MNIST (SPECIALIZED STORES)")
    print("="*100)
    fashion_mnist_noniid_results = safe_experiment_runner(
        run_fashion_mnist_federated_experiment,
        iid=False,
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
        max_local_epochs=MAX_LOCAL_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE
    )
    """

    if fashion_mnist_results:
        print("\n🎉 Enhanced Fashion-MNIST Federated Learning Research Complete!")
        print("📊 Results demonstrate DLA performance on fashion recognition tasks")
        print("👗 Fashion-MNIST provides ideal complexity for evaluating partial training strategies")
        print("🏪 Realistic federated scenario: Fashion retailers with diverse inventories")

        # Save results to JSON
        print("\n💾 Saving results...")
        serializable_results = []
        for result in fashion_mnist_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    serializable_result[key] = {str(k): (v.tolist() if hasattr(v, 'tolist') else v)
                                              for k, v in value.items()}
                elif hasattr(value, 'tolist'):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)

        with open('./results/fashion_mnist_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print("💾 Results saved to ./results/fashion_mnist_results.json")
        print("📊 Plots saved to ./plots/ directory")
        print("✅ Experiment completed successfully!")
    else:
        print("❌ Experiment failed!")
        sys.exit(1)

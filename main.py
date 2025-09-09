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

# =================== GLOBAL EARLY STOPPING FRAMEWORK ===================

class GlobalConvergenceTracker:
    """
    Research-grade global convergence tracking with early stopping
    """
    def __init__(self, dataset_name, patience=15, accuracy_plateau=0.1, min_rounds=50):
        self.dataset_name = dataset_name
        self.patience = patience
        self.accuracy_plateau = accuracy_plateau
        self.min_rounds = min_rounds
        
        # Dataset-specific convergence thresholds
        self.convergence_thresholds = {
            'MNIST': 95.0,        # Easy dataset - high threshold
            'Fashion-MNIST': 85.0, # Moderate dataset - medium threshold  
            'CIFAR-10': 75.0      # Hard dataset - lower threshold
        }
        
        self.min_accuracy = self.convergence_thresholds.get(dataset_name, 85.0)
        
        # Tracking variables
        self.no_improvement_count = 0
        self.best_accuracy = 0.0
        self.convergence_round = None
        self.converged = False
        self.accuracy_history = []
        
    def update_accuracy(self, current_accuracy, round_num):
        """Update accuracy and check for convergence"""
        self.accuracy_history.append(current_accuracy)
        
        # Don't check convergence until minimum rounds
        if round_num < self.min_rounds:
            return False
            
        # Check for improvement
        if current_accuracy > self.best_accuracy + self.accuracy_plateau:
            self.best_accuracy = current_accuracy
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            
        # Check convergence criteria
        accuracy_threshold_met = current_accuracy >= self.min_accuracy
        plateau_reached = self.no_improvement_count >= self.patience
        
        if accuracy_threshold_met and plateau_reached and not self.converged:
            self.convergence_round = round_num
            self.converged = True
            return True
            
        return False
        
    def get_convergence_info(self):
        """Get comprehensive convergence information"""
        return {
            'converged': self.converged,
            'convergence_round': self.convergence_round,
            'best_accuracy': self.best_accuracy,
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0,
            'min_accuracy_threshold': self.min_accuracy,
            'rounds_to_convergence': self.convergence_round if self.converged else None,
            'accuracy_at_convergence': self.best_accuracy if self.converged else None
        }

# =================== RESEARCH-GRADE JSON SERIALIZATION ===================

def safe_serialize_value(value):
    """Recursively convert values to JSON-safe types (Research Standard)"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [safe_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): safe_serialize_value(v) for k, v in value.items()}
    elif isinstance(value, defaultdict):
        return {str(k): safe_serialize_value(v) for k, v in dict(value).items()}
    elif isinstance(value, set):
        return list(value)
    return value

def prepare_results_for_json(results):
    """Convert all results to JSON-serializable format (Publication Standard)"""
    return {str(key): safe_serialize_value(value) for key, value in results.items()}

# =================== ENHANCED STRAGGLER TRACKING ===================

class ResearchGradeStragglerTracker:
    """Research-grade straggler tracking following FL literature best practices"""
    def __init__(self, num_clients, client_speeds, straggler_threshold=1.5):
        self.num_clients = num_clients
        self.client_speeds = client_speeds
        self.straggler_threshold = straggler_threshold

        # Track stragglers by capability vs performance
        self.round_stragglers = []
        self.client_straggler_history = {cid: [] for cid in range(num_clients)}
        self.total_stragglers_per_round = []

        # Enhanced participation tracking
        self.participation_metrics = {
            'slow_capable_clients': [i for i, speed in enumerate(client_speeds) if speed < 0.8],
            'normal_capable_clients': [i for i, speed in enumerate(client_speeds) if 0.8 <= speed <= 1.2],
            'fast_capable_clients': [i for i, speed in enumerate(client_speeds) if speed > 1.2],
        }

        self.accommodation_history = []

    def identify_round_stragglers_with_participation(self, client_times, round_num):
        """Enhanced straggler identification with participation analysis"""
        median_time = float(np.median(client_times))
        threshold = median_time * self.straggler_threshold

        round_stragglers = []
        for cid, time_taken in enumerate(client_times):
            is_straggler = time_taken > threshold
            round_stragglers.append(is_straggler)
            self.client_straggler_history[cid].append(is_straggler)

        straggler_count = int(sum(round_stragglers))
        self.round_stragglers.append(round_stragglers)
        self.total_stragglers_per_round.append(straggler_count)

        participation_analysis = self.analyze_participation_capability(client_times, threshold)
        self.accommodation_history.append(participation_analysis)

        return round_stragglers, straggler_count, participation_analysis

    def analyze_participation_capability(self, client_times, threshold):
        """Analyze algorithm's ability to accommodate vs exclude stragglers"""
        slow_clients = self.participation_metrics['slow_capable_clients']
        normal_clients = self.participation_metrics['normal_capable_clients']
        fast_clients = self.participation_metrics['fast_capable_clients']

        accommodation_threshold = threshold * 1.2

        successful_slow = sum(1 for cid in slow_clients if client_times[cid] <= accommodation_threshold)
        successful_normal = sum(1 for cid in normal_clients if client_times[cid] <= threshold)
        successful_fast = sum(1 for cid in fast_clients if client_times[cid] <= threshold)

        total_slow = len(slow_clients)
        total_normal = len(normal_clients)
        total_fast = len(fast_clients)

        return {
            'total_clients': self.num_clients,
            'slow_capable_clients': total_slow,
            'normal_capable_clients': total_normal,
            'fast_capable_clients': total_fast,
            'successful_slow_participation': successful_slow,
            'successful_normal_participation': successful_normal,
            'successful_fast_participation': successful_fast,
            'slow_inclusion_rate': successful_slow / max(total_slow, 1),
            'normal_inclusion_rate': successful_normal / max(total_normal, 1),
            'fast_inclusion_rate': successful_fast / max(total_fast, 1),
            'stragglers_accommodated': successful_slow,
            'stragglers_excluded': total_slow - successful_slow,
            'accommodation_efficiency': successful_slow / max(total_slow, 1)
        }

    def get_comprehensive_straggler_statistics(self):
        """Research-grade comprehensive straggler statistics"""
        if not self.total_stragglers_per_round:
            return {}

        avg_stragglers = float(np.mean(self.total_stragglers_per_round))
        straggler_rate = avg_stragglers / self.num_clients

        if self.accommodation_history:
            avg_accommodation = float(np.mean([h['accommodation_efficiency'] for h in self.accommodation_history]))
            avg_slow_inclusion = float(np.mean([h['slow_inclusion_rate'] for h in self.accommodation_history]))
        else:
            avg_accommodation = 0.0
            avg_slow_inclusion = 0.0

        persistent_stragglers = self.get_persistent_stragglers()

        return {
            'avg_stragglers_per_round': avg_stragglers,
            'max_stragglers_in_round': int(max(self.total_stragglers_per_round)),
            'min_stragglers_in_round': int(min(self.total_stragglers_per_round)),
            'straggler_rate': straggler_rate,
            'persistent_stragglers': persistent_stragglers,
            'straggler_variance': float(np.var(self.total_stragglers_per_round)),
            'avg_accommodation_efficiency': avg_accommodation,
            'avg_slow_inclusion_rate': avg_slow_inclusion,
            'total_slow_devices': len(self.participation_metrics['slow_capable_clients']),
            'accommodation_success_score': avg_accommodation * avg_slow_inclusion
        }

    def get_persistent_stragglers(self, persistence_threshold=0.7):
        """Identify clients that are frequently stragglers"""
        persistent = []
        for cid, history in self.client_straggler_history.items():
            if len(history) > 0:
                rate = sum(history) / len(history)
                if rate >= persistence_threshold:
                    persistent.append(int(cid))
        return persistent

# =================== ENHANCED EFFICIENCY TRACKING ===================

class ResearchGradeEfficiencyTracker:
    """Research-grade efficiency tracking for MLP federated learning"""
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.client_flops = defaultdict(list)
        self.client_memory_usage = defaultdict(list)
        self.client_energy_estimate = defaultdict(list)
        self.client_communication_costs = defaultdict(list)
        self.total_communication_per_round = []
        self.communication_breakdown = defaultdict(list)
        self.convergence_efficiency = {}
        self.parameter_efficiency = defaultdict(list)

    def start_client_monitoring(self, client_id):
        """Start monitoring computational resources"""
        return {
            'start_time': time.time(),
            'start_memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }

    def calculate_mlp_training_flops(self, model, batch_size, active_layers_count):
        """Calculate accurate MLP training FLOPs following ML literature standards"""
        total_flops = 0
        linear_layers = [module for module in model.model.modules() if isinstance(module, nn.Linear)]
        active_layers = linear_layers[:active_layers_count] if active_layers_count <= len(linear_layers) else linear_layers

        for layer in active_layers:
            forward_flops = batch_size * layer.in_features * layer.out_features
            backward_flops = 2 * forward_flops
            layer_flops = forward_flops + backward_flops
            total_flops += layer_flops

        return int(total_flops)

    def calculate_federated_communication_cost(self, strategy, full_model_state_dict, active_layers_count):
        """Calculate bidirectional communication cost with protocol overhead"""
        full_model_size = 0
        for param in full_model_state_dict.values():
            full_model_size += param.numel() * 4

        active_model_size = 0
        linear_layers = [name for name in full_model_state_dict.keys()
                        if 'weight' in name or 'bias' in name]

        total_layers = len([name for name in linear_layers if 'weight' in name])
        if total_layers > 0:
            layers_to_include = min(active_layers_count, total_layers)
            layers_to_skip = max(0, total_layers - layers_to_include)

            layer_count = 0
            for name, param in full_model_state_dict.items():
                if 'weight' in name:
                    if layer_count >= layers_to_skip:
                        active_model_size += param.numel() * 4
                        bias_name = name.replace('weight', 'bias')
                        if bias_name in full_model_state_dict:
                            active_model_size += full_model_state_dict[bias_name].numel() * 4
                    layer_count += 1

        if strategy == "fedavg":
            downstream_bytes = full_model_size
            upstream_bytes = full_model_size
        elif strategy in ["fedpmt", "feddrop", "dla"]:
            downstream_bytes = full_model_size
            upstream_bytes = active_model_size
        else:
            downstream_bytes = full_model_size
            upstream_bytes = full_model_size

        protocol_overhead = (downstream_bytes + upstream_bytes) * 0.05
        total_communication = downstream_bytes + upstream_bytes + protocol_overhead

        return {
            'downstream_bytes': int(downstream_bytes),
            'upstream_bytes': int(upstream_bytes),
            'protocol_overhead': int(protocol_overhead),
            'total_bytes': int(total_communication),
            'full_model_size': int(full_model_size),
            'active_model_size': int(active_model_size)
        }

    def end_client_monitoring(self, client_id, start_metrics, model_state_dict,
                             layer_depth, strategy, batch_size=64):
        """End monitoring and calculate comprehensive efficiency metrics"""
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        training_time = end_time - start_metrics['start_time']
        memory_used = abs(end_memory - start_metrics['start_memory'])

        temp_model = MNISTNet()
        flops = self.calculate_mlp_training_flops(temp_model, batch_size, layer_depth)
        comm_cost = self.calculate_federated_communication_cost(strategy, model_state_dict, layer_depth)
        estimated_energy = training_time * 40.0

        self.client_flops[client_id].append(flops)
        self.client_memory_usage[client_id].append(int(memory_used))
        self.client_energy_estimate[client_id].append(estimated_energy)
        self.client_communication_costs[client_id].append(comm_cost['total_bytes'])

        return {
            'training_time': training_time,
            'flops': flops,
            'communication_cost': comm_cost,
            'energy': estimated_energy,
            'memory_used': int(memory_used)
        }

# =================== MNIST MODEL ARCHITECTURE ===================

class MNISTNet(nn.Module):
    """Research-grade MNIST neural network architecture"""
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128, 64], num_classes=10, dropout_rate=0.3):
        super(MNISTNet, self).__init__()
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def apply_layer_mask(model, layers_to_update):
    """Apply mask to update only the last 'layers_to_update' layers"""
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

# =================== ENHANCED TRAINING FUNCTIONS ===================

def train_locally_with_monitoring(model, train_loader, val_loader, layers_to_update, device,
                                 client_speed=1.0, max_epochs=15, lr=0.001, patience=5):
    """Enhanced local training with comprehensive monitoring"""
    model = copy.deepcopy(model).to(device)
    apply_layer_mask(model, layers_to_update)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    start_time = time.time()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []
    val_accuracies = []

    model.train()
    for epoch in range(max_epochs):
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

            if client_speed < 1.0:
                time.sleep(0.001 * (1.0 - client_speed))

        avg_train_loss = epoch_train_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

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

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_accuracy = 100 * correct / max(total, 1)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        model.train()

        if patience_counter >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    elapsed_time = (time.time() - start_time) / client_speed
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_accuracy = val_accuracies[-1] if val_accuracies else 0
    epochs_trained = len(train_losses)

    return (model.state_dict(), elapsed_time, final_train_loss,
            final_val_accuracy, epochs_trained, train_losses, val_losses, val_accuracies)

# =================== DLA IMPLEMENTATION ===================

def adjust_layers_improved(elapsed_time, current_depth, client_id,
                         T_low=8.0, T_high=25.0, max_layers=5):
    """Improved layer adjustment with thresholds optimized for MNIST simplicity"""
    client_T_low = T_low * (0.7 + client_id * 0.1)
    client_T_high = T_high * (0.8 + client_id * 0.2)

    if elapsed_time > client_T_high and current_depth > 1:
        return current_depth - 1
    elif elapsed_time < client_T_low and current_depth < max_layers:
        return current_depth + 1
    else:
        return current_depth

class DLAStabilityTracker:
    """Enhanced DLA stability tracking"""
    def __init__(self, num_clients):
        self.adaptation_history = {cid: [] for cid in range(num_clients)}
        self.consecutive_suggestions = {cid: {'increase': 0, 'decrease': 0, 'stay': 0}
                                      for cid in range(num_clients)}
        self.performance_history = {cid: [] for cid in range(num_clients)}

    def should_adapt(self, client_id, current_depth, suggested_depth,
                    client_performance=None, stability_threshold=3):
        """Enhanced adaptation decision"""
        if client_performance is not None:
            self.performance_history[client_id].append(client_performance)

        if suggested_depth == current_depth:
            self.consecutive_suggestions[client_id] = {'increase': 0, 'decrease': 0, 'stay': 0}
            return current_depth

        if suggested_depth > current_depth:
            self.consecutive_suggestions[client_id]['increase'] += 1
            self.consecutive_suggestions[client_id]['decrease'] = 0
            self.consecutive_suggestions[client_id]['stay'] = 0

            if self.consecutive_suggestions[client_id]['increase'] >= stability_threshold:
                self.consecutive_suggestions[client_id]['increase'] = 0
                return suggested_depth

        elif suggested_depth < current_depth:
            self.consecutive_suggestions[client_id]['decrease'] += 1
            self.consecutive_suggestions[client_id]['increase'] = 0
            self.consecutive_suggestions[client_id]['stay'] = 0

            if self.consecutive_suggestions[client_id]['decrease'] >= stability_threshold:
                self.consecutive_suggestions[client_id]['decrease'] = 0
                return suggested_depth

        return current_depth

# =================== EVALUATION AND AGGREGATION ===================

def evaluate_model(model, test_loader, device):
    """Enhanced model evaluation"""
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

    accuracy = 100 * correct / max(total, 1)
    avg_loss = total_loss / max(len(test_loader), 1)
    return float(accuracy), float(avg_loss)

def average_weights(weight_list, client_sizes=None):
    """Weighted average of model weights"""
    if client_sizes is None:
        client_sizes = [1] * len(weight_list)

    total_size = sum(client_sizes)
    avg_weights = copy.deepcopy(weight_list[0])

    for key in avg_weights:
        avg_weights[key] = torch.zeros_like(avg_weights[key])

    for i, weights in enumerate(weight_list):
        weight = client_sizes[i] / total_size
        for key in avg_weights:
            avg_weights[key] += weights[key] * weight

    return avg_weights

# =================== FEDERATED DATASET CREATION ===================

def create_federated_mnist(num_clients=50, batch_size=64, iid=True, val_split=0.2):
    """Create research-grade federated MNIST dataset"""
    print("Downloading MNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    total_samples = len(train_dataset)
    base_samples_per_client = total_samples // num_clients

    clients = []
    client_speeds = []

    print(f"Creating {num_clients} federated MNIST clients...")

    for client_id in range(num_clients):
        if client_id < num_clients * 0.1:
            speed_factor = np.random.uniform(0.3, 0.6)
        elif client_id < num_clients * 0.2:
            speed_factor = np.random.uniform(1.8, 2.5)
        else:
            speed_factor = np.random.uniform(0.8, 1.4)

        client_samples = base_samples_per_client

        if iid:
            indices = np.random.choice(range(total_samples), client_samples, replace=False)
        else:
            num_categories = np.random.randint(2, 5)
            target_classes = np.random.choice(range(10), size=num_categories, replace=False)
            indices = []
            targets = np.array(train_dataset.targets)

            for target_class in target_classes:
                class_indices = np.where(targets == target_class)[0]
                class_samples = min(len(class_indices), client_samples // len(target_classes))
                selected = np.random.choice(class_indices, class_samples, replace=False)
                indices.extend(selected)

            indices = indices[:client_samples]

            if client_id < 5:
                client_specialization = [mnist_classes[i] for i in target_classes]
                print(f"Client {client_id} specializes in digits: {client_specialization}")

        client_dataset = Subset(train_dataset, indices)
        train_size = int((1 - val_split) * len(client_dataset))
        val_size = len(client_dataset) - train_size

        train_subset, val_subset = random_split(client_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        clients.append((train_loader, val_loader, len(client_dataset)))
        client_speeds.append(speed_factor)

    print(f"Created {num_clients} MNIST clients with {total_samples} total samples")
    return clients, test_loader, client_speeds

# =================== RESEARCH-GRADE FEDERATED TRAINING WITH GLOBAL EARLY STOPPING ===================

def research_grade_federated_training_mnist_with_early_stopping(clients, client_speeds, max_rounds, device, 
                                                               strategy="fedavg", max_layers=5, fixed_depth=3, 
                                                               test_loader=None, lr=0.001, max_local_epochs=15):
    """
    Research-grade federated training for MNIST with GLOBAL EARLY STOPPING
    """
    global_model = MNISTNet().to(device)

    # Initialize research-grade tracking
    straggler_tracker = ResearchGradeStragglerTracker(len(clients), client_speeds)
    efficiency_tracker = ResearchGradeEfficiencyTracker(len(clients))
    
    # GLOBAL EARLY STOPPING TRACKER
    convergence_tracker = GlobalConvergenceTracker('MNIST', patience=15, accuracy_plateau=0.1, min_rounds=50)

    # Comprehensive results tracking
    results = {
        'test_accuracies': [],
        'test_losses': [],
        'training_times': defaultdict(list),
        'client_depths': defaultdict(list),
        'client_epochs_trained': defaultdict(list),
        'client_val_accuracies': defaultdict(list),
        'round_times': [],
        'losses': defaultdict(list),
        'convergence_info': {},
        'stragglers_per_round': [],
        'client_straggler_history': defaultdict(list),
        'straggler_statistics': {},
        'participation_analysis': [],
        'flops_per_round': [],
        'communication_costs_per_round': [],
        'energy_consumption_per_round': [],
        'computational_efficiency': [],
        'communication_efficiency': [],
        'client_communication_breakdown': defaultdict(list),
        # CONVERGENCE TRACKING
        'rounds_to_convergence': None,
        'flops_until_convergence': 0,
        'communication_until_convergence': 0,
        'energy_until_convergence': 0,
        'accuracy_at_convergence': None,
        'early_stopped': False
    }

    # Initialize DLA components
    current_depths = {cid: 3 for cid in range(len(clients))}
    dla_tracker = DLAStabilityTracker(len(clients)) if strategy == "dla" else None

    print(f"\nRESEARCH-GRADE MNIST FEDERATED LEARNING WITH GLOBAL EARLY STOPPING")
    print(f"Strategy: {strategy.upper()}")
    print(f"Clients: {len(clients)}")
    print(f"Max Rounds: {max_rounds} | Early Stop Threshold: 95.0% | Patience: 15 rounds")
    print(f"Comprehensive Tracking: Enabled")
    print("-" * 80)

    for r in range(max_rounds):
        round_start_time = time.time()
        print(f"\n--- Round {r+1}/{max_rounds} ---")

        local_weights = []
        local_sizes = []
        round_training_times = []
        round_flops = []
        round_comm_costs = []
        round_energy = []

        # Process each client
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

            results['client_depths'][cid].append(layer_depth)

            # Start comprehensive monitoring
            start_metrics = efficiency_tracker.start_client_monitoring(cid)

            # Local training
            training_results = train_locally_with_monitoring(
                global_model, train_loader, val_loader, layer_depth, device,
                client_speed=client_speeds[cid], max_epochs=max_local_epochs, lr=lr, patience=5
            )

            (local_state_dict, elapsed_time, final_train_loss,
             final_val_accuracy, epochs_trained, train_losses, val_losses, val_accuracies) = training_results

            # End comprehensive monitoring
            efficiency_metrics = efficiency_tracker.end_client_monitoring(
                cid, start_metrics, local_state_dict, layer_depth, strategy
            )

            # Record comprehensive metrics
            results['training_times'][cid].append(elapsed_time)
            results['losses'][cid].append(final_train_loss)
            results['client_epochs_trained'][cid].append(epochs_trained)
            results['client_val_accuracies'][cid].append(final_val_accuracy)
            results['client_communication_breakdown'][cid].append(efficiency_metrics['communication_cost'])

            round_training_times.append(elapsed_time)
            round_flops.append(efficiency_metrics['flops'])
            round_comm_costs.append(efficiency_metrics['communication_cost']['total_bytes'])
            round_energy.append(efficiency_metrics['energy'])

            # Show detailed progress for first few clients
            if cid < 3:
                comm = efficiency_metrics['communication_cost']
                print(f"Client {cid:2d}: depth={layer_depth}, time={elapsed_time:5.1f}s, "
                      f"FLOPs={efficiency_metrics['flops']/1e6:.1f}M, "
                      f"Comm={comm['total_bytes']/1024:.1f}KB "
                      f"(↓{comm['downstream_bytes']/1024:.1f}KB + ↑{comm['upstream_bytes']/1024:.1f}KB), "
                      f"Energy={efficiency_metrics['energy']:.1f}J")

            # DLA depth adaptation (MNIST-specific thresholds)
            if strategy == "dla":
                suggested_depth = adjust_layers_improved(elapsed_time, layer_depth, cid,
                                                       T_low=8.0, T_high=25.0, max_layers=max_layers)
                new_depth = dla_tracker.should_adapt(cid, layer_depth, suggested_depth,
                                                    client_performance=final_val_accuracy, stability_threshold=3)
                current_depths[cid] = new_depth

            local_weights.append(local_state_dict)
            local_sizes.append(dataset_size)

        # COMPREHENSIVE STRAGGLER AND PARTICIPATION ANALYSIS
        round_stragglers, straggler_count, participation_analysis = straggler_tracker.identify_round_stragglers_with_participation(
            round_training_times, r
        )

        # Record round-level comprehensive metrics
        results['stragglers_per_round'].append(straggler_count)
        results['participation_analysis'].append(participation_analysis)
        results['flops_per_round'].append(int(np.sum(round_flops)))
        results['communication_costs_per_round'].append(int(np.sum(round_comm_costs)))
        results['energy_consumption_per_round'].append(float(np.sum(round_energy)))

        # Calculate efficiency metrics
        if results['test_accuracies']:
            prev_acc = results['test_accuracies'][-1]
        else:
            prev_acc = 0

        current_flops = np.sum(round_flops)
        current_comm = np.sum(round_comm_costs)

        comp_efficiency = prev_acc / max(current_flops / 1e9, 1e-10)
        comm_efficiency = prev_acc / max(current_comm / 1e6, 1e-10)

        results['computational_efficiency'].append(comp_efficiency)
        results['communication_efficiency'].append(comm_efficiency)

        for cid, is_straggler in enumerate(round_stragglers):
            results['client_straggler_history'][cid].append(is_straggler)

        # Comprehensive round summary
        straggler_ids = [cid for cid, is_straggler in enumerate(round_stragglers) if is_straggler]
        median_time = float(np.median(round_training_times))
        avg_flops = float(np.mean(round_flops)) / 1e6
        total_comm = float(np.sum(round_comm_costs)) / 1024
        total_energy = float(np.sum(round_energy))

        print(f"\nRound {r+1} MNIST Research-Grade Analysis:")
        print(f"   Median Training Time: {median_time:.1f}s")
        print(f"   Stragglers: {straggler_count}/{len(clients)} ({straggler_count/len(clients)*100:.1f}%)")
        print(f"   Slow Device Inclusion: {participation_analysis['slow_inclusion_rate']*100:.1f}%")
        print(f"   Accommodation Success: {participation_analysis['accommodation_efficiency']*100:.1f}%")
        print(f"   Total MLP FLOPs: {avg_flops*len(clients):.1f}M")
        print(f"   Total Communication: {total_comm:.1f}KB")
        print(f"   Total Energy: {total_energy:.1f}J")
        if straggler_ids:
            print(f"   Straggler IDs: {straggler_ids[:5]}{'...' if len(straggler_ids) > 5 else ''}")

        # Federated aggregation
        global_weights = average_weights(local_weights, local_sizes)
        global_model.load_state_dict(global_weights)

        # Global model evaluation
        if test_loader:
            test_acc, test_loss = evaluate_model(global_model, test_loader, device)
            results['test_accuracies'].append(test_acc)
            results['test_losses'].append(test_loss)

            print(f"   Global Test Accuracy: {test_acc:.2f}%")
            if current_flops > 0:
                efficiency_score = test_acc / max(avg_flops, 1e-10)
                print(f"   MLP Efficiency: {efficiency_score:.2f} acc/MFLOP")

            # CHECK FOR GLOBAL CONVERGENCE
            converged = convergence_tracker.update_accuracy(test_acc, r + 1)
            if converged:
                print(f"   *** GLOBAL CONVERGENCE DETECTED at Round {r+1} ***")
                print(f"   *** Accuracy: {test_acc:.2f}% >= Threshold: {convergence_tracker.min_accuracy}% ***")
                print(f"   *** Stopping early to save resources ***")
                
                # Record convergence metrics
                results['rounds_to_convergence'] = r + 1
                results['flops_until_convergence'] = int(np.sum(results['flops_per_round']))
                results['communication_until_convergence'] = int(np.sum(results['communication_costs_per_round']))
                results['energy_until_convergence'] = float(np.sum(results['energy_consumption_per_round']))
                results['accuracy_at_convergence'] = test_acc
                results['early_stopped'] = True
                break

        round_time = time.time() - round_start_time
        results['round_times'].append(round_time)

        # Progress indicator every 25 rounds
        if (r + 1) % 25 == 0:
            print(f"\nProgress: {r+1}/{max_rounds} rounds completed")
            if results['test_accuracies']:
                current_acc = results['test_accuracies'][-1]
                print(f"   Current MNIST Accuracy: {current_acc:.2f}%")
                print(f"   Target for Early Stop: {convergence_tracker.min_accuracy}%")

    # Final comprehensive statistics
    convergence_info = convergence_tracker.get_convergence_info()
    results['convergence_info'] = convergence_info
    
    if not convergence_info['converged']:
        print(f"\n*** No convergence achieved within {max_rounds} rounds ***")
        results['rounds_to_convergence'] = max_rounds
        results['flops_until_convergence'] = int(np.sum(results['flops_per_round']))
        results['communication_until_convergence'] = int(np.sum(results['communication_costs_per_round']))
        results['energy_until_convergence'] = float(np.sum(results['energy_consumption_per_round']))
        results['accuracy_at_convergence'] = results['test_accuracies'][-1] if results['test_accuracies'] else 0.0
        results['early_stopped'] = False

    results['straggler_statistics'] = straggler_tracker.get_comprehensive_straggler_statistics()

    return global_model, results, straggler_tracker, efficiency_tracker

# =================== RESEARCH-GRADE ANALYSIS FUNCTIONS ===================

def analyze_comprehensive_research_performance_mnist_early_stop(all_results, strategies, straggler_trackers, efficiency_trackers):
    """Research-grade comprehensive analysis for MNIST MLP federated learning with early stopping"""
    print(f"\n{'='*80}")
    print("RESEARCH-GRADE MNIST MLP PERFORMANCE ANALYSIS WITH GLOBAL EARLY STOPPING")
    print("Digit Recognition Federated Learning with Convergence Efficiency")
    print(f"{'='*80}")

    analysis_summary = {}

    for i, strategy in enumerate(strategies):
        results = all_results[i]
        straggler_stats = results.get('straggler_statistics', {})

        if not results.get('test_accuracies'):
            continue

        # Core Performance Metrics
        final_accuracy = float(results['test_accuracies'][-1])
        convergence_info = results.get('convergence_info', {})
        rounds_to_convergence = results.get('rounds_to_convergence', len(results['test_accuracies']))
        early_stopped = results.get('early_stopped', False)
        
        # Efficiency metrics until convergence
        flops_until_convergence = results.get('flops_until_convergence', 0)
        comm_until_convergence = results.get('communication_until_convergence', 0)
        energy_until_convergence = results.get('energy_until_convergence', 0.0)
        
        # Time efficiency
        total_time = float(np.sum(results.get('round_times', [])[:rounds_to_convergence]))

        # Research-Grade Straggler Metrics
        stragglers_until_convergence = int(np.sum(results.get('stragglers_per_round', [])[:rounds_to_convergence]))
        straggler_rate = straggler_stats.get('straggler_rate', 0)
        accommodation_efficiency = straggler_stats.get('avg_accommodation_efficiency', 0)
        slow_inclusion_rate = straggler_stats.get('avg_slow_inclusion_rate', 0)

        # Calculate convergence efficiency ratios
        convergence_computational_efficiency = final_accuracy / max(flops_until_convergence / 1e9, 1e-10)
        convergence_communication_efficiency = final_accuracy / max(comm_until_convergence / 1e6, 1e-10)
        convergence_energy_efficiency = final_accuracy / max(energy_until_convergence, 1e-10)
        convergence_time_efficiency = final_accuracy / max(total_time, 1e-10)

        analysis_summary[strategy] = {
            'final_accuracy': final_accuracy,
            'rounds_to_convergence': rounds_to_convergence,
            'early_stopped': early_stopped,
            'convergence_time': total_time,
            'flops_until_convergence': flops_until_convergence,
            'communication_until_convergence': comm_until_convergence,
            'energy_until_convergence': energy_until_convergence,
            'stragglers_until_convergence': stragglers_until_convergence,
            'straggler_rate': straggler_rate,
            'accommodation_efficiency': accommodation_efficiency,
            'slow_inclusion_rate': slow_inclusion_rate,
            'convergence_computational_efficiency': convergence_computational_efficiency,
            'convergence_communication_efficiency': convergence_communication_efficiency,
            'convergence_energy_efficiency': convergence_energy_efficiency,
            'convergence_time_efficiency': convergence_time_efficiency
        }

        convergence_status = "CONVERGED EARLY" if early_stopped else "NO CONVERGENCE"
        
        print(f"\n{strategy.upper()} - MLP Convergence Analysis:")
        print(f"   Final Accuracy: {final_accuracy:.2f}%")
        print(f"   Convergence Status: {convergence_status}")
        print(f"   Rounds to Convergence: {rounds_to_convergence}")
        print(f"   Time to Convergence: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   FLOPs until Convergence: {flops_until_convergence/1e9:.2f} GFLOP")
        print(f"   Communication until Convergence: {comm_until_convergence/1e6:.2f} MB")
        print(f"   Energy until Convergence: {energy_until_convergence:.1f} J")
        print(f"   Stragglers until Convergence: {stragglers_until_convergence}")
        print(f"   Slow Device Inclusion: {slow_inclusion_rate*100:.1f}%")
        print(f"   CONVERGENCE Computational Efficiency: {convergence_computational_efficiency:.4f} acc/GFLOP")
        print(f"   CONVERGENCE Communication Efficiency: {convergence_communication_efficiency:.4f} acc/MB")
        print(f"   CONVERGENCE Time Efficiency: {convergence_time_efficiency:.4f} acc/s")

    return analysis_summary

def plot_convergence_analysis_mnist(all_results, strategies):
    """Research-grade convergence visualization for MNIST"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MNIST MLP Federated Learning - Global Early Stopping Analysis\n(Convergence Efficiency Comparison)',
                 fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    line_styles = ['-', '--', '-.', ':']

    # 1. Accuracy Evolution with Convergence Points
    axes[0, 0].set_title('MNIST Accuracy Evolution\n(Early Stopping Points)', fontsize=12, fontweight='bold')
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies']:
            rounds = range(1, len(results['test_accuracies'])+1)
            axes[0, 0].plot(rounds, results['test_accuracies'],
                           label=f'{strategy.upper()}', color=colors[i],
                           linewidth=2.5, linestyle=line_styles[i])
            
            # Mark convergence point
            if results.get('early_stopped', False):
                conv_round = results.get('rounds_to_convergence', len(results['test_accuracies']))
                conv_acc = results['test_accuracies'][conv_round-1] if conv_round <= len(results['test_accuracies']) else results['test_accuracies'][-1]
                axes[0, 0].scatter(conv_round, conv_acc, color=colors[i], s=100, marker='*', edgecolor='black', linewidth=1)
                
    # Add convergence threshold line
    axes[0, 0].axhline(y=95.0, color='red', linestyle='--', alpha=0.7, label='Convergence Threshold (95%)')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Rounds to Convergence Comparison
    axes[0, 1].set_title('Rounds to Convergence\n(Efficiency Metric)', fontsize=12, fontweight='bold')
    conv_rounds = []
    strategy_names = []
    colors_used = []
    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        rounds = results.get('rounds_to_convergence', 200)
        conv_rounds.append(rounds)
        strategy_names.append(strategy.upper())
        colors_used.append(colors[i])

    bars = axes[0, 1].bar(strategy_names, conv_rounds, color=colors_used, alpha=0.8)
    axes[0, 1].set_ylabel('Rounds to Convergence')
    for bar, val, result in zip(bars, conv_rounds, all_results):
        label = f'{val}' if result.get('early_stopped', False) else f'{val}*'
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(conv_rounds)*0.01,
                       label, ha='center', va='bottom', fontweight='bold')

    # 3. Convergence Communication Efficiency
    axes[0, 2].set_title('Communication Efficiency\n(Until Convergence)', fontsize=12, fontweight='bold')
    comm_eff = []
    for strategy, results in zip(strategies, all_results):
        final_acc = results['test_accuracies'][-1] if results['test_accuracies'] else 0
        total_comm = results.get('communication_until_convergence', 1)
        eff = final_acc / max(total_comm / 1e6, 1e-10)
        comm_eff.append(eff)

    bars = axes[0, 2].bar(strategy_names, comm_eff, color=colors_used, alpha=0.8)
    axes[0, 2].set_ylabel('Accuracy / MB (until convergence)')
    for bar, val in zip(bars, comm_eff):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comm_eff)*0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

    # 4. Convergence Computational Efficiency
    axes[1, 0].set_title('Computational Efficiency\n(Until Convergence)', fontsize=12, fontweight='bold')
    comp_eff = []
    for strategy, results in zip(strategies, all_results):
        final_acc = results['test_accuracies'][-1] if results['test_accuracies'] else 0
        total_flops = results.get('flops_until_convergence', 1)
        eff = final_acc / max(total_flops / 1e9, 1e-10)
        comp_eff.append(eff)

    bars = axes[1, 0].bar(strategy_names, comp_eff, color=colors_used, alpha=0.8)
    axes[1, 0].set_ylabel('Accuracy / GFLOP (until convergence)')
    for bar, val in zip(bars, comp_eff):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comp_eff)*0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

    # 5. Energy Efficiency Until Convergence
    axes[1, 1].set_title('Energy Efficiency\n(Until Convergence)', fontsize=12, fontweight='bold')
    energy_eff = []
    for strategy, results in zip(strategies, all_results):
        final_acc = results['test_accuracies'][-1] if results['test_accuracies'] else 0
        total_energy = results.get('energy_until_convergence', 1)
        eff = final_acc / max(total_energy, 1e-10)
        energy_eff.append(eff)

    bars = axes[1, 1].bar(strategy_names, energy_eff, color=colors_used, alpha=0.8)
    axes[1, 1].set_ylabel('Accuracy / Joule (until convergence)')
    for bar, val in zip(bars, energy_eff):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energy_eff)*0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

    # 6. Convergence Summary Table
    axes[1, 2].set_title('MNIST Convergence Summary', fontsize=12, fontweight='bold')
    summary_text = []
    for strategy, results in zip(strategies, all_results):
        if results['test_accuracies']:
            final_acc = results['test_accuracies'][-1]
            rounds = results.get('rounds_to_convergence', 200)
            early_stop = "✓" if results.get('early_stopped', False) else "✗"
            
            summary_text.append(
                f"{strategy.upper()}:\n"
                f"  Accuracy: {final_acc:.1f}%\n"
                f"  Rounds: {rounds}\n"
                f"  Early Stop: {early_stop}\n"
            )

    axes[1, 2].text(0.05, 0.95, '\n'.join(summary_text), transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show("mnist_convergence_analysis")

# =================== MAIN RESEARCH EXPERIMENT ===================

def run_research_grade_mnist_experiment_early_stop(iid=True, num_rounds=300,
                                                   num_clients=50, max_local_epochs=15):
    """Run research-grade MNIST MLP experiment with global early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create federated MNIST dataset
    data_type = "IID" if iid else "Non-IID"
    print(f"Creating federated MNIST dataset ({data_type})...")
    clients, test_loader, client_speeds = create_federated_mnist(
        num_clients=num_clients, batch_size=64, iid=iid, val_split=0.2
    )

    strategies = ["fedavg", "fedpmt", "feddrop", "dla"]
    all_results = []
    all_straggler_trackers = []
    all_efficiency_trackers = []

    print(f"\n{'='*80}")
    print(f"RESEARCH-GRADE MNIST MLP FEDERATED LEARNING WITH GLOBAL EARLY STOPPING")
    print(f"Best Practices: Convergence Detection, Efficiency-to-Convergence Analysis")
    print(f"Max Rounds: {num_rounds} | Early Stop: 95.0% accuracy threshold")
    print(f"Dataset: MNIST (10 digit classes, 28x28 grayscale images)")
    print(f"Architecture: 4-layer MLP (784-512-256-128-64-10)")
    print(f"{'='*80}")

    # Run experiments for each strategy
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Running {strategy.upper()} - MLP with Global Early Stopping")
        print(f"{'='*80}")

        set_seed(42)  # Reset seed for fair comparison

        start_time = time.time()
        global_model, results, straggler_tracker, efficiency_tracker = research_grade_federated_training_mnist_with_early_stopping(
            clients=clients,
            client_speeds=client_speeds,
            max_rounds=num_rounds,
            device=device,
            strategy=strategy,
            max_layers=5,
            fixed_depth=3,
            test_loader=test_loader,
            lr=0.001,
            max_local_epochs=max_local_epochs
        )

        experiment_time = time.time() - start_time
        all_results.append(results)
        all_straggler_trackers.append(straggler_tracker)
        all_efficiency_trackers.append(efficiency_tracker)

        # Convergence summary
        if results['test_accuracies']:
            final_acc = results['test_accuracies'][-1]
            rounds_to_conv = results.get('rounds_to_convergence', num_rounds)
            early_stopped = results.get('early_stopped', False)
            conv_status = "CONVERGED EARLY" if early_stopped else "NO CONVERGENCE"

            print(f"\n{strategy.upper()} CONVERGENCE RESULTS:")
            print(f"   Final Accuracy: {final_acc:.2f}%")
            print(f"   Convergence Status: {conv_status}")
            print(f"   Rounds to Convergence: {rounds_to_conv}")
            print(f"   Experiment Time: {experiment_time/60:.1f} minutes")

    # Generate convergence analysis
    print(f"\n{'='*80}")
    print("GENERATING CONVERGENCE EFFICIENCY ANALYSIS")
    print(f"{'='*80}")

    analysis_summary = analyze_comprehensive_research_performance_mnist_early_stop(
        all_results, strategies, all_straggler_trackers, all_efficiency_trackers
    )
    plot_convergence_analysis_mnist(all_results, strategies)

    return all_results, all_straggler_trackers, all_efficiency_trackers, analysis_summary

# =================== CLUSTER ENVIRONMENT SETUP ===================

def setup_research_environment():
    """Setup research-grade environment for MNIST"""
    plt.ioff()

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")

    os.makedirs('./data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    print("Research environment setup complete")

def save_plot_for_publication():
    """Replace plt.show() with publication-quality saving"""
    original_show = plt.show

    def publication_show(filename_prefix="plot"):
        timestamp = int(time.time())
        filename = f"./plots/{filename_prefix}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Publication-quality plot saved: {filename}")
        plt.close()

    plt.show = publication_show

# =================== MAIN EXECUTION ===================

if __name__ == "__main__":
    setup_research_environment()
    save_plot_for_publication()
    
    NUM_CLIENTS = 50
    MAX_LOCAL_EPOCHS = 15
    NUM_ROUNDS = 300  # Higher max since we expect early stopping

    print("Starting Research-Grade MNIST Federated Learning with Global Early Stopping")
    print(f"Max Rounds: {NUM_ROUNDS} | Early Stop Target: 95.0% accuracy")
    print(f"Clients: {NUM_CLIENTS} | Strategies: FedAvg, FedPMT, FedDrop, DLA")

    # Run research-grade experiment with early stopping
    experiment_results = run_research_grade_mnist_experiment_early_stop(
        iid=True,
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
        max_local_epochs=MAX_LOCAL_EPOCHS
    )

    if experiment_results:
        all_results, straggler_trackers, efficiency_trackers, analysis_summary = experiment_results

        print("\nResearch-Grade MNIST Early Stopping Experiment Complete!")
        print("Convergence efficiency analysis generated!")

        # Save results
        print("\nSaving convergence analysis results...")
        try:
            serializable_results = []
            for result in all_results:
                serializable_result = prepare_results_for_json(result)
                serializable_results.append(serializable_result)

            research_data = {
                'experiment_results': serializable_results,
                'analysis_summary': prepare_results_for_json(analysis_summary),
                'experiment_config': {
                    'max_rounds': NUM_ROUNDS,
                    'num_clients': NUM_CLIENTS,
                    'max_local_epochs': MAX_LOCAL_EPOCHS,
                    'early_stopping': True,
                    'convergence_threshold': 95.0,
                    'patience': 15,
                    'strategies': ["fedavg", "fedpmt", "feddrop", "dla"],
                    'dataset': 'MNIST',
                    'architecture': 'MLP'
                },
                'metadata': {
                    'timestamp': time.time(),
                    'convergence_framework': 'Global Early Stopping',
                    'best_practices_applied': [
                        'Global early stopping with convergence detection',
                        'Dataset-specific convergence thresholds',
                        'Efficiency-to-convergence metrics',
                        'Comprehensive convergence tracking',
                        'Realistic federated learning scenarios'
                    ]
                }
            }

            with open('./results/mnist_early_stop_results.json', 'w') as f:
                json.dump(research_data, f, indent=2)

            print("Results saved to ./results/mnist_early_stop_results.json")
            print("Publication-quality plots saved to ./plots/ directory")
            print("MNIST Global Early Stopping experiment completed successfully!")

        except Exception as e:
            print(f"Warning: Could not save JSON results due to: {e}")
            print("Experiment completed successfully, but results not saved to file")

    else:
        print("MNIST early stopping experiment failed!")
        sys.exit(1)

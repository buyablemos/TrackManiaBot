import time
import torch
from src.horizon.dqn.model import Model, Trainer
from src.config import Config


def benchmark_model_performance(num_iterations=1000):
    """
    Benchmark the inference and training speed of the model.
    :param num_iterations: Number of iterations to run the benchmark for
    :return: Dictionary containing benchmark results
    """
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")

    # Initialize model
    model = Model(device, Config.DQN.NUMBER_OF_QUANTILES, Config.DQN.N_COS, False, False).to(device)
    trainer = Trainer(model, device, Config.DQN.LEARNING_RATE, Config.DQN.GAMMA)

    # Create random input for inference benchmark
    single_input = torch.rand((1, Config.Arch.INPUT_SIZE), device=device)

    # Benchmark inference speed
    print("\n=== Inference Speed Benchmark ===")
    start_time = time.time()

    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(single_input)

    inference_time = time.time() - start_time
    print(f"Inference time for {num_iterations} iterations: {inference_time:.4f} seconds")
    print(f"Average inference time per sample: {inference_time / num_iterations * 1000:.4f} ms")
    print(f"Inference throughput: {num_iterations / inference_time:.2f} samples/second")

    # Benchmark training speed
    print("\n=== Training Speed Benchmark ===")

    start_time = time.time()

    for i in range(num_iterations):
        state = torch.rand(Config.Arch.INPUT_SIZE, device=device)
        action = model(state)
        reward = torch.rand(1, device=device)
        next_state = torch.rand(Config.Arch.INPUT_SIZE, device=device)
        done = False
        trainer.train_step(state, action, reward, next_state, done)

    training_time = time.time() - start_time
    print(f"Training time for {num_iterations}: {training_time:.4f} seconds")
    print(f"Average training time per step: {training_time / num_iterations * 1000:.4f} ms")
    print(f"Training throughput: {num_iterations / training_time:.2f} samples/second")

    # Model size and memory usage
    model_size = sum(p.numel() for p in model.parameters())
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB

    print("\n=== Model Information ===")
    print(f"Model parameters: {model_size:,}")
    print(f"Model memory footprint: {model_memory:.2f} MB")

    return {
        "device": str(device),
        "inference_time_per_sample_ms": inference_time / num_iterations * 1000,
        "inference_throughput": num_iterations / inference_time,
        "training_time_per_step_ms": training_time / num_iterations * 1000,
        "training_throughput": num_iterations / training_time,
        "model_parameters": model_size,
        "model_memory_mb": model_memory
    }


def calculate_max_game_speed(benchmark_results):
    """
    Calculate the theoretical maximum game speed based on benchmark results
    """
    # Get the training time per step in ms
    train_time_ms = benchmark_results["training_time_per_step_ms"]

    # Game takes actions every 100ms
    game_action_interval_ms = Config.Game.INTERVAL_BETWEEN_ACTIONS

    # Calculate max speed (with 80% safety buffer)
    theoretical_max_speed = game_action_interval_ms / train_time_ms
    safe_max_speed = theoretical_max_speed * 0.8

    print(f"Training step takes {train_time_ms:.2f}ms")
    print(f"Theoretical maximum game speed: {theoretical_max_speed:.1f}x")
    print(f"Recommended maximum game speed: {safe_max_speed:.1f}x")

    return safe_max_speed

if __name__ == "__main__":
    results = benchmark_model_performance()
    print("\nBenchmark completed!")
    print("\n=== Speed Recommendations ===")
    max_speed = calculate_max_game_speed(results)

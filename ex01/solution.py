
# Imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch


PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Convenience functions
def plot_model(model=None):
    # Visualize data
    plt.plot(torch.linspace(0, 1, 1000), ground_truth_function(torch.linspace(0, 1, 1000)), label='Ground truth')
    plt.plot(x_train, y_train, 'ob', label='Train data')
    plt.plot(x_test, y_test, 'xr', label='Test data')
    # Visualize model
    if model is not None:
        plt.plot(torch.linspace(0, 1, 1000), model(torch.linspace(0, 1, 1000)), label=f'Model of degree: {model.degree()}')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    

# Generate data
n_samples = 11
noise_amplitude = 0.15

def ground_truth_function(x):
    # Generate data of the form sin(2 * Pi * x)
    result = torch.sin(2 * np.pi * x)
    return result

torch.manual_seed(42)

x_test = torch.linspace(0, 1, n_samples)
y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(n_samples,))
x_train = torch.linspace(0, 1, n_samples)
y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(n_samples,))

# Test plotting
plot_model()
plt.savefig(PLOTS_DIR / "Initial_data.png")
plt.clf()


# Model fitting

def error_function(model, x_data, y_data):
    y_pred = model(x_data.numpy())
    error = 0.5 * np.sum((y_pred - y_data.numpy()) ** 2)
    return error

model_degree = 3

model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
train_err = error_function(model, x_train, y_train)
test_err = error_function(model, x_test, y_test)

print(f"{train_err=}, {test_err=}")

# Result plotting
plot_model(model)
plt.savefig(PLOTS_DIR / 'Initial_fit.png')
plt.clf()

# ---- Continue with the exercises on the degree of the polynomial and the exploration of data size


# Degree-11 polynomial fitting
model_degree = 11

model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)

train_err = error_function(model, x_train, y_train)
test_err = error_function(model, x_test, y_test)

print(f"Degree 11: {train_err=}, {test_err=}")

plot_model(model)
plt.savefig(PLOTS_DIR / "Degree_11_fit.png", dpi=300, bbox_inches="tight")
plt.close()

def rms_error(model, x_data, y_data):
    # RMS = sqrt(2 * squared_error / number_of_samples)
    error = error_function(model, x_data, y_data)
    return np.sqrt(2 * error / len(x_data))


# Polynomial degree vs RMS error

degrees = range(0, 12)
train_errors = []
test_errors = []

for degree in degrees:
    model = np.polynomial.Polynomial.fit(
        x_train, y_train, deg=degree
    )

    train_errors.append(rms_error(model, x_train, y_train))
    test_errors.append(rms_error(model, x_test, y_test))

plt.figure(figsize=(8, 5))

plt.plot(degrees, train_errors, "o-", label="Training RMS error")
plt.plot(degrees, test_errors, "s-", label="Test RMS error")

plt.xlabel("Polynomial degree")
plt.ylabel("RMS error")
plt.title("Polynomial Degree vs Training and Test RMS Error")
plt.xticks(range(0, 12))
plt.legend()
plt.grid(True)

plt.savefig(PLOTS_DIR / "Degree_vs_RMS_error.png", dpi=300)
plt.close()

# Experiment: Effect of training dataset size on overfitting

model_degree = 10
sample_sizes = [10, 11, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

train_rms_errors = []
test_rms_errors = []

# Keep the test dataset fixed across all experiments
torch.manual_seed(123)
x_test_large = torch.linspace(0, 1, 1000)
y_test_large = ground_truth_function(x_test_large) + torch.normal(
    0., noise_amplitude, size=(1000,)
)

for n in sample_sizes:
    # Generate training data with n samples
    torch.manual_seed(42)
    x_train_new = torch.linspace(0, 1, n)
    y_train_new = ground_truth_function(x_train_new) + torch.normal(
        0., noise_amplitude, size=(n,)
    )

    # Fit a polynomial of fixed degree 10
    model = np.polynomial.Polynomial.fit(
        x_train_new.numpy(),
        y_train_new.numpy(),
        deg=model_degree
    )

    train_rms = rms_error(model, x_train_new, y_train_new)
    test_rms = rms_error(model, x_test_large, y_test_large)

    train_rms_errors.append(train_rms)
    test_rms_errors.append(test_rms)

    print(f"N={n:4d} | Train RMS={train_rms:.4f} | Test RMS={test_rms:.4f}")

# Plot sample size against RMS error
plt.figure(figsize=(9, 5))

plt.plot(sample_sizes, train_rms_errors, "o-", label="Training RMS error")
plt.plot(sample_sizes, test_rms_errors, "s-", label="Test RMS error")

plt.xlabel("Number of training samples")
plt.ylabel("RMS error")
plt.title("Effect of Dataset Size on Overfitting (Degree 10)")
plt.legend()
plt.grid(True)

plt.savefig(PLOTS_DIR / "Sample_size_vs_RMS_error.png", dpi=300, bbox_inches="tight")
plt.close()
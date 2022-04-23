import matplotlib.pyplot as plt
import numpy as np

def plot_budgets(budgets, accs, ylabel="Accuracy (%)", p=None):
  plt.plot(budgets, accs)
  plt.xlabel("Hidden Size")
  plt.ylabel(ylabel)
  plt.title(f"Inference Hidden Size vs. {ylabel}")

  idx = np.argmax(accs)
  best = accs[idx]
  plt.plot(idx, best, "ro", label=f"Highest accuracy: {best}%")

  median = np.percentile(accs,50,interpolation="nearest")
  plt.plot(accs.index(median), median, "go", label=f"Median accuracy: {median}%")

  if p is not None:
    plt.axvline(x=p * max(budgets), label="Average training hidden size")

  plt.legend()

  plt.savefig("budget.png")
  plt.clf()

def plot_timing(budgets, times, ylabel="Time (s)"):
  plt.plot(budgets, times)
  plt.xlabel("Hidden Size")
  plt.ylabel(ylabel)
  plt.title(f"Inference Hidden Size vs. {ylabel}")

  plt.savefig("times.png")
  plt.clf()

def plot_number_parameters(budgets, n_params, ylabel="Total Model Parameters"):
  plt.plot(budgets, n_params)
  plt.plot(
    [budgets[0], budgets[-1]],
    [n_params[0], n_params[-1]],
    label="linear increase",
  )
  plt.xlabel("Hidden Size")
  plt.ylabel(ylabel)
  plt.title(f"Inference Hidden Size vs. {ylabel}")

  plt.savefig("param_count.png")
  plt.legend()
  plt.clf()

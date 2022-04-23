import matplotlib.pyplot as plt
import numpy as np

def plot_budgets(budgets, accs, ylabel="Accuracy (%)"):
  plt.plot(budgets, accs)
  plt.xlabel("Hidden Size")
  plt.ylabel(ylabel)
  plt.title(f"Inference Hidden Size vs. {ylabel}")
  plt.savefig("budget.png")

  idx = np.argmax(accs)
  best = accs[idx]
  plt.plot(idx, best, "ro", label="Highest accuracy")

  median = np.percentile(accs,50,interpolation='nearest')
  plt.plot(accs.index(median), median, "go", label="Median accuracy")

  plt.legend()
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
  plt.xlabel("Hidden Size")
  plt.ylabel(ylabel)
  plt.title(f"Inference Hidden Size vs. {ylabel}")
  plt.savefig("param_count.png")
  plt.clf()

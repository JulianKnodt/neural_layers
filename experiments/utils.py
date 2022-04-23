import matplotlib.pyplot as plt

def plot_budgets(budgets, accs, ylabel="Accuracy (%)"):
  plt.plot(budgets, accs)
  plt.xlabel("Hidden Size")
  plt.ylabel(ylabel)
  plt.title(f"Inference Hidden Size vs. {ylabel}")
  plt.savefig("budget.png")

def plot_timing(budgets, times, ylabel="Time (s)"):
  plt.plot(budgets, times)
  plt.xlabel("Hidden Size")
  plt.ylabel(ylabel)
  plt.title(f"Inference Hidden Size vs. {ylabel}")
  plt.savefig("times.png")

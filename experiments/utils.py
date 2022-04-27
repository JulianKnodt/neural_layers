import matplotlib.pyplot as plt
import numpy as np

def plot_budgets(budgets, accs, ylabel="Accuracy (%)", p=None, title="PointNet"):
  budgets = list(budgets)
  plt.plot(budgets, accs)
  plt.xlabel("Hidden Size")
  plt.ylabel(ylabel)
  plt.title(f"{title}: Hidden Size vs. {ylabel}")

  idx = np.argmax(accs)
  best = accs[idx]
  plt.plot(budgets[idx], best, "ro", label=f"Highest Accuracy: {100*best:.02f}%, Hidden Size: {budgets[idx]}")

  qtr = np.percentile(accs,25,interpolation="nearest")
  plt.plot(
    budgets[accs.index(qtr)], qtr, "yo",
    label=f"Lower Quartile Accuracy: {100*qtr:.02f}%, Hidden Size: {budgets[accs.index(qtr)]}",
  )

  median = np.percentile(accs,50,interpolation="nearest")
  plt.plot(
    budgets[accs.index(median)], median, "go",
    label=f"Median Accuracy: {100*median:.02f}%, Hidden Size: {budgets[accs.index(median)]}",
  )

  if p is not None:
    plt.axvline(
      x=(0.5 + p/2) * max(budgets),
      color="k", linestyle="--",
      label="Average training cutoff",
    )

  plt.legend()

  plt.savefig("budget.png", pad_inches=0.15)
  plt.clf()

def plot_number_parameters(budgets, n_params, ylabel="Total Parameters", title="PointNet"):
  budgets = list(budgets)
  plt.plot(budgets, n_params, label="Model Parameters")
  plt.plot(
    [budgets[0], budgets[-1]],
    [n_params[0], n_params[-1]],
    "g--",
    label="Linear Increase (Reference)",
  )
  plt.xlabel("Hidden Size")

  plt.ticklabel_format(style="sci", axis="y")
  plt.ylabel(ylabel)

  plt.title(f"{title}: Hidden Size vs. {ylabel}")

  mid = len(budgets)//4
  plt.plot(budgets[mid], n_params[mid], "yo", label=f"Lower Quartile Parameter Count: {n_params[mid]}")

  mid = len(budgets)//2
  plt.plot(budgets[mid], n_params[mid], "go", label=f"Median Parameter Count: {n_params[mid]}")

  plt.plot(budgets[-1], n_params[-1], "ro", label=f"Maximum Parameter Count: {n_params[-1]}")

  plt.legend()

  plt.savefig("param_count.png", pad_inches=0.15)
  plt.clf()


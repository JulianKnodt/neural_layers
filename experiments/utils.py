import matplotlib.pyplot as plt

def plot_budgets(budgets, accs):
  plt.plot(budgets, accs)
  plt.xlabel("budgets")
  plt.ylabel("accuracy")
  plt.title("latent budget relative to total accuracy")
  plt.savefig("budget.png")

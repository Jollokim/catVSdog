import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import style


# style.use("ggplot")

model_name = "model-catVSdog-1641761042"

df = pd.read_csv(model_name + ".log", names=["epoch", "acc", "val-acc"])

print(df)

plt.plot(df["epoch"], df["acc"], label="acc")
plt.plot(df["epoch"], df["val-acc"], label="val-acc")
plt.show()
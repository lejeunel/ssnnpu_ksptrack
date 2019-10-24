import pandas as pd
import matplotlib.pyplot as plt


path = '/home/krakapwa/otlshare/laurent.lejeune/medical-labeling/Dataset24/precomp_descriptors/log_train.csv'
df = pd.read_csv(path)

plt.plot(df.values.ravel()[1:])
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

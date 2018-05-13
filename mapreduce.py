import numpy as np
import pandas as pd

array = np.random.randint(5, size=(10, 3))

mapper = lambda x: x ** 2
vfun = np.vectorize(mapper)

"""for x in np.nditer(array):
	print (x)"""

index = ['primero', 'segundo', 'tercero']
df = pd.DataFrame(data=array, columns=index)
df2 = df.groupby(['primero'])
df3 = df.groupby(['segundo'])
df4 = df.groupby(['tercero'])

df2_count = df2.primero.count()
df3_count = df3.segundo.count()
df4_count = df4.tercero.count()

df2_mult = df2_count * 0.7
df3_mult = df3_count * 0.2
df4_mult = df4_count * 0.1

print(df2_mult)
print(df3_mult)
print(df4_mult)
# print(df3.segundo.count().get_values() * 0.20)
# print(df3.segundo.keys())

conatenated = pd.concat([df2_mult, df3_mult, df4_mult], axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=0, names=None, verify_integrity=False,
          copy=True)

print(conatenated)
d = {}
for i, j in conatenated.iteritems():
	if d.get(i) is not None:
		d[i] += j
	else:
		d[i] = j

print(d)

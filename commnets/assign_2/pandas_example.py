# http://stackoverflow.com/questions/11640243/pandas-plot-multiple-y-axes

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
df = DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])

fig, ax = plt.subplots()
#ax2, ax3 = ax.twinx(), ax.twinx()
ax2 = ax.twinx()
#rspine = ax3.spines['right']
#rspine.set_position(('axes', 1.25))
#ax3.set_frame_on(True)
#ax3.patch.set_visible(False)
fig.subplots_adjust(right=0.75)

df.A.plot(ax=ax, style='b-')
df.B.plot(ax=ax2, style='r-', secondary_y=True)
#df.C.plot(ax=ax3, style='g-')


plt.show()

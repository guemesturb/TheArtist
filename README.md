# The Artist
Plotting class preferences created by guemesturb.

# Usage examples
The API con be invoked as in the next snippet.
```python
from TheArtist import TheArtist
import numpy as np


x = np.arange(1,10)
y = np.arange(1,20)
z = np.sin(x)
figure = TheArtist(latex=False,n_rows = 2,n_cols = 1)
figure.plot(x, y, 0, 0, color='darkblue', linewidth=1, linestyle='-', marker='o')
figure.scatter(x, z, 0, 1, marker='s')
figure.set_labels(['Eje 1', 'Eje V'], 0, 0)
figure.set_labels(['Eje 2', None], 0, 1)
figure.savefig("test", fig_format='pdf', dots_per_inch=1000)
````
It is possible to acces the Matplotlib figure accesing its property and apply other functions or methods:
```python
figure.fig.savefig("test.png") # savefig method from Matplotlib figure
```
More examples can be found in the examples directory. 
# Installing TheArtist
You can install it using PyPi: 
```
pip install TheArtist
```
It is possible to install it from source using the code below:
```shell
git clone https://github.com/guemesturb/TheArtist/tree/develop 
cd TheArtist
pip install .
```

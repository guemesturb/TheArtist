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
More examples can be found in the examples directory. 
# Installing TheArtist
Future versions may be uploaded to PyPi. At the moment, to install the current version, you may use the code below.
```shell
pip install wheel
git clone https://github.com/guemesturb/TheArtist/tree/develop 
cd TheArtist
python3 setup.py sdist bdist_wheel
conda activate your_env
pip install .
```

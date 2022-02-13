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
figure = TheArtist(latex=False)
figure.generate_figure_environment(cols=1, rows=2, fig_width_pt=384, ratio='golden', regular=True)
figure.plot_lines(x, y, 0, 0, color='darkblue', linewidth=1, linestyle='-', marker='o')
figure.plot_scatter(x, z, 0, 1, marker='s')
figure.set_labels(['Eje 1', 'Eje V'], 0, 0)
figure.set_labels(['Eje 2', None], 0, 0)
figure.savefig("test", fig_format='pdf', dots_per_inch=1000)
````
More examples can be found in the examples directory. 
# Installing TheArtist
Future versions may be uploaded to PyPi. At the moment, to install the current version, you may use the code below.
```shell
pip install wheel
git clone https://github.com/guemesturb/TheArtist.git
cd TheArtist
python3 setup.py sdist bdist_wheel
conda activate your_env
pip install .
```

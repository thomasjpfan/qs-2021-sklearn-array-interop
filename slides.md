title: Scikit-learn Array Interop
use_katex: False
class: title-slide

# Scikit-learn Array Interop

![](images/scikit-learn-logo-notext.png)

.larger[Thomas J. Fan]<br>
@thomasjpfan<br>
<a class="this-talk-link", href="https://github.com/thomasjpfan/qs-2021-sklearn-array-interop" target="_blank">
This talk on Github: thomasjpfan/qs-2021-sklearn-array-interop</a>

---

.g[
.g-6[
# Three Options So Far

1. `__array_function__` [NEP-18](https://numpy.org/neps/nep-0018-array-function-protocol.html)
2. `__array_module__` [NEP-37](https://numpy.org/neps/nep-0037-array-module.html)
3. `uarray`
]
.g-6[
# Limitations
- No Cython or complied code
- To support JAX: No inplace operations
]
]

---

# `__array_function__`

- Was BC breaking, Scikit-learn now always cast to numpy array: [#14702](https://github.com/scikit-learn/scikit-learn/pull/14702)
- As with all other options: requires a switch to turn it on

---

# `__array_function__`

- Was explored by folks at Vaex: [#16196](https://github.com/scikit-learn/scikit-learn/pull/16196)
- Uses `np.empty(like=)` for array creation.
- In combination with `__array_ufunc__` + explicit checks for methods: (`np.mean`), mostly works.
- **Limitations**: Can not call into `scipy.linalg`

---

# `__array_module__`

- Gives more control to sklearn
- I explored this with PCA + cupy: [#16574](https://github.com/scikit-learn/scikit-learn/pull/16574)
- Unable to get `scipy` module
    - Most likely need scipy to support `__array_module__`
    - Implementation depends on the mechanism `scipy` chooses to turn on `array_module`

---

# `__array_module__` requires switch

### Global config
```python
*sklearn.set_config(array_module=True)

X = cupy.random.randn(4, 10)
scalar = StandardScalar()

# X_trans is a cupy array
X_trans = scalar.fit_transform(X)
```

### Context manager
```python
X = cupy.random.randn(4, 10)
scalar = StandardScalar()

*with sklearn.config_context(array_module=True):
# X_trans is a cupy array
    X_trans = scalar.fit_transform(X)
```

---

# `__array_module__`

.g.g-middle[
.g-6[
### Object that holds state
```python
scalar = StandardScalar(array_module=True)

# X_trans is a cupy array
X_trans = scalar.fit_transform(X)
```
]
.g-6[
### Functions

```python
scale(..., array_module=True)
```
]
]


---

# Global config requires some care

- When spawning process -> global config must be passed to new processes: [#17634](https://github.com/scikit-learn/scikit-learn/pull/17634)
- When spawning threads -> changes to global config in one thread should not alter other threads: [#17634](https://github.com/scikit-learn/scikit-learn/pull/18736)

---

# `uarray`

- User story in [NEP 31](https://numpy.org/neps/nep-0031-uarray.html) makes sense.

---

# API
## If `numpy.scipy.linalg` is in the namespace

```python
import uarray as ua
import cupy

ua.register_backend(cupy)

X = cupy.random.randn(100, 10)
pca = PCA()

# X_trans is a cupy array
X_trans = pca.fit_transform(X)
```

---

# API?
## If `numpy.scipy.linalg` is not in the namespace

```python
import uarray as ua
import cupy
import scipy

ua.register_backend(cupy)

X = cupy.random.randn(100, 10)
pca = PCA()

with scipy.fft.set_backend(cu_fft)
    # X_trans is a cupy array
    X_trans = pca.fit_transform(X)
```

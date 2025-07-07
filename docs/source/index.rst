Welcome to elementary's documentation!
===================================

Generic differentiable accelerator elements modeling in JAX.
Single particle hamiltonian function:

.. math::
   :nowrap:

   \begin{eqnarray}
   & H(q_x, q_y, q_s, p_x, p_y, p_s; s) = \frac{p_s}{\beta} - t(s)(q_x p_y - q_y p_x) - (1 + h(s) q_x) \left(\sqrt{P_s^2 - P_x^2 - P_y^2 - \frac{1}{\beta^2 \gamma^2}} + a_s(q_x, q_y, q_s; s)\right)  \\
   & \\
   & P_s = p_s + 1/\beta - \varphi(q_x, q_y, q_s; s)  \\
   & P_x = p_x - a_x(q_x, q_y, q_s; s)  \\
   & P_y = p_y - a_y(q_x, q_y, q_s; s)
   \end{eqnarray}

where :math:`\beta` and :math:`\gamma` are the relativistic factors, :math:`h(s)` is the reference trajectory curvature and :math:`t(s)` is the reference trajectory torsion, :math:`a_x(q_x, q_y, q_s; s)`, :math:`a_y(q_x, q_y, q_s; s)` and :math:`a_s(q_x, q_y, q_s; s)` are the scaled vector potential components, and :math:`\varphi(q_x, q_y, q_s; s)` is the scaled scalar potential.
Additionaly, longitudinal coordinate and momentum are given by:

.. math::
  :nowrap:

   \begin{eqnarray}
   & q_s = \frac{s}{\beta} - c t \\
   & p_s = \frac{E}{c P} - \frac{1}{\beta}
   \end{eqnarray}

Common predefined elements are available or you can create your own by specifying scaled potentials and reference trajectory parameters (curvature and torsion).
All but vector potentials arguments are optional for hamiltonian and element construction.
Vector and scalar potentials are assumed to have matching signatures.

.. code-block:: python

   def vector(qs:Array, s:Array, *args:Array) -> tuple[Array, Array, Array]:
      q_x, q_y, q_s = qs
      ...

   def scalar(qs:Array, s:Array, *args:Array) -> Array:
      q_x, q_y, q_s = qs
      ...

This is also the case for curvature and torsion functions.
Note, same extra arguments as in vector and scalar functions should be passed.

.. code-block:: python

   def curvature(s:Array, *args:Array) -> Array:
      ...

   def torsion(s:Array, *args:Array) -> Array:
      ...

The resulting hamiltonian and element signatures are:

.. code-block:: python

   def hamiltonian(qs: Array, ps: Array, s: Array, *args: Array) -> Array:
      q_x, q_y, q_s = qs
      p_x, p_y, p_s = ps
      ...

   def element(qsps:Array, length:Array, start:Array, *args:Array) -> Array:
      qs, ps = jax.numpy.reshape(qsps, (2, -1))
      q_x, q_y, q_s = qs
      p_x, p_y, p_s = ps
      ...

.. toctree::
   :caption: Examples:
   :maxdepth: 1

   examples/example-00.ipynb
   examples/example-01.ipynb
   examples/example-02.ipynb
   examples/example-03.ipynb
   examples/example-04.ipynb
   examples/example-05.ipynb
   examples/example-06.ipynb
   examples/example-07.ipynb
   examples/example-08.ipynb
   examples/example-09.ipynb
   examples/example-10.ipynb
   examples/example-11.ipynb
   examples/example-12.ipynb

   examples/wm-01.ipynb
   examples/wm-02.ipynb
   examples/wm-03.ipynb
   examples/wm-04.ipynb
   examples/wm-05.ipynb
   examples/wm-06.ipynb
   examples/wm-07.ipynb

.. toctree::
   :caption: API:
   :maxdepth: 1

   modules/util.rst
   modules/hamiltonian.rst
   modules/element.rst
   modules/drift.rst
   modules/quadrupole.rst
   modules/sextupole.rst
   modules/octupole.rst
   modules/multipole.rst
   modules/dipole.rst
   modules/alignment.rst
   modules/cavity.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

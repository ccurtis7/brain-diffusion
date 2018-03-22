.. _doc-label:

Documentation
=============

.. _msd-label:

MSD
---

MSD calculations are carried out using vectorized numpy arrays.  Calculations
are carried out according to the formula <MSD> = (x-x0)^2 + (y-y0)^2.  Video-wide
geometric averages are also reported.

.. _hyak-label:

Hyak
----

MSD calculations can be parallelized using a computer cluster.  An example
implementation is provided `here <https://github.com/ccurtis7/brain_diffusion/blob/master/brain_diffusion/hyak_msd_template.py>`_
for use on the Shared Scalable Computer Cluster for Research
`(Hyak) <http://wiki.cac.washington.edu/display/hyakusers/WIKI+for+Hyak+users>`_
at the University of Washington.

.. _visualization-label:

Visualization
-------------

Current visualization tools include a histogram plot of MSD/Diffusion distributions.

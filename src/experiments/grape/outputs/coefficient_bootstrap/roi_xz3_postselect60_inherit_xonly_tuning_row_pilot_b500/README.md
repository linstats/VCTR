# ROI X+Z3 post-selection row-bootstrap pilot

Exploratory ROI paired VCTR fit using VF00, VF04, and VF60, selected from the
prior unpenalized 60Z ROI analysis by nominal significance. The model inherits
the ROI X-only tuning and uses `B=500` paired-row bootstrap replicates.

Because variable selection used the same outcome data and the bootstrap treats
repeated visits from a patient as independent rows, these intervals are not
selection-adjusted or patient-cluster-valid manuscript inference.

This directory contains the complete three-variable beta table, coefficient
and variance summaries, aggregation metadata, and ROI `A(t)` and `sigma(t)`
figures.

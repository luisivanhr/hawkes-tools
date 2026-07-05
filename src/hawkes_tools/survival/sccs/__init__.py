"""SCCS learner variants for standalone hawkes-tools survival workflows."""

from hawkes_tools.survival.convolutional_sccs import BatchConvSCCS, StreamConvSCCS

__all__ = ["BatchConvSCCS", "StreamConvSCCS"]

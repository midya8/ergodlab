"""Exact stationary policy and reward audits."""
from .core import audit
from .discount import audit_discounted
from .covariance import audit_covariance
__all__=['audit','audit_discounted','audit_covariance']

"""Validator-parity evaluation for Teutonic SN3.

Everything that decides a duel -- dataset planning, sequence sampling, per-sequence
cross-entropy numerics and the paired-bootstrap verdict -- is imported from the live
validator package (pinned in :mod:`src.parity.upstream`). This package only adds what
the validator does not need: multi-GPU layer pipelining for 32 GB cards, weight
merging at load time, loss caching and submission-risk diagnostics.
"""

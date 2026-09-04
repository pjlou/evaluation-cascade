# src/discovery
#
# Cascade Stage 2 of the verification cascade: statistical / embedding-based failure
# discovery. Unlike src/verifiers (Cascade Stage 1: rule-graph-grounded,
# forced-choice), this module works on free-form generation and doesn't require a gold label —
# it finds candidate failure PATTERNS via clustering, for a human to inspect
# and, where warranted, promote into a new deterministic rule graph node (Cascade Stage 1).
#
# See README.md ("Cascade Stage 2: Statistical Failure Discovery") for the full loop.

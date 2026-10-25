"""docproof: field extraction with validation, calibrated confidence and human review.

The package is organised as a pipeline of small modules (schemas, validators, grounding,
confidence, routing). Each stage is deterministic over its inputs; anything that touches a
model, a file or the network lives at the edges so the core can be tested without them.
"""

__version__ = "0.0.1"

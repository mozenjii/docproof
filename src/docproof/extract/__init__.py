"""Extractors: turn an OCR page into candidate field values with the text they came from.

Every extractor returns candidates, never decisions. Grounding, validation and routing decide what
happens to each candidate (``docproof.pipeline``), so extractors can be swapped and compared.
"""

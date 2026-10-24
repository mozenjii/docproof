# docproof

Extracts structured fields from invoices and freight documents, checks them against rules, and sends only the doubtful fields to a human.

> **Status:** early. I'm at the design stage and there is no released code yet. The results table below stays empty until a script in this repo produces the numbers. Only public and synthetic documents are used here.

## The problem

Small businesses type invoice and shipping document fields into spreadsheets by hand. Extraction demos work on clean scans and fall over on a skewed fax, and nobody can say which fields are safe to accept without a look. A wrong amount costs far more than a second glance, so the useful number isn't accuracy. It's how many fields you can accept automatically at a precision you can defend.

## What it will do

- Extract fields (vendor, dates, line items, totals, references) from scans and PDFs.
- Check every value: totals add up, dates make sense, vendors match a list, and the value is actually present on the page, with a bounding box to prove it.
- Combine those signals into a calibrated confidence per field and auto-accept only above a threshold chosen for a target precision.
- Show the rest in a review screen that highlights the evidence, and keep reviewer corrections as new training and test data.
- Compare approaches on the same held-out set: OCR plus a language model, a vision model, and a small model fine-tuned with LoRA.

## How I plan to know it works

Data comes from public receipt and form datasets plus a synthetic generator for invoices and freight documents. The generator emits labels together with each document, varies the layout and degrades the scans. The honest test for any model is layouts it never saw during training, so some templates are held out entirely.

| Measurement | How | Result |
|---|---|---|
| Field-level F1 | Public and synthetic held-out sets | not measured |
| Prompted baseline vs fine-tuned small model: accuracy, cost per document, latency | Identical held-out layouts | not measured |
| Share of fields auto-accepted at a fixed precision | Calibrated confidence on held-out documents | not measured |
| Accuracy on degraded scans | Degraded copies of the same documents | not measured |

## Design notes

**Grounding as a hallucination check.** A value that can't be found on the page is wrong or invented. Every extracted value has to be located in the OCR output, and the box it came from is what the reviewer sees.

**Ground truth by construction.** Generating documents and labels together removes the annotation bottleneck. I'll check the synthetic set against real public scans to make sure it isn't easier than the real thing.

**Fine-tune only if it earns it.** The fine-tuned model has to beat a strong prompted baseline, including constrained JSON decoding, on cost or accuracy at the same review rate. If it doesn't, that goes in the write-up.

**Straight-through rate is the headline.** Not accuracy, but the fraction of fields that skip review at a precision target.

## Roadmap

- [ ] Field schema, validators and scorer
- [ ] Synthetic document generator with scan degradation
- [ ] Extraction baselines
- [ ] Confidence calibration and review queue
- [ ] LoRA experiment on held-out layouts
- [ ] Write-up

## Related projects

Part of a set of six projects about building LLM systems you can trust:
[evalgate](https://github.com/mozenjii/evalgate),
[policylens](https://github.com/mozenjii/policylens),
[docproof](https://github.com/mozenjii/docproof),
[tollgate](https://github.com/mozenjii/tollgate),
[redline](https://github.com/mozenjii/redline),
[casefile](https://github.com/mozenjii/casefile).

Built by Mohib Ahmad.

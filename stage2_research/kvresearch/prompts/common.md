# KVResearch isolated agent contract

Read every listed input file directly from disk. Do not rely on conversational memory.
Write exactly the requested artifact to the absolute output path. Preserve the user request and all locked constraints.
Control JSON may contain only its domain fields; never emit the reserved envelope keys `schema_version`, `artifact_type`, `run_id`, `created_at`, `producer`, `inputs`, or `status`. The deterministic host adds those fields and authoritative input references. JSONL output must contain one valid JSON object per line.
Never invent experimental numbers, file paths, citations, or successful checks. Reference evidence by absolute path and SHA-256 when available.
Do not modify either VecInfer baseline directory. Candidate changes belong only in the isolated workspace under the run directory.
Do not return the artifact only in chat. Write it to the exact output path before finishing.

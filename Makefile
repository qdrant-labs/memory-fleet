# Fleet Memory — build/run targets

.PHONY: setup lint run run-b reset demo-check demo-restore demo-save demo-scale

setup:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

run:
	uv run python -m fleetmemory.server

run-b:
	uv run python -m fleetmemory.server --instance b

reset:
	rm -rf edge-data

# --- demo rituals ---

# offline preflight: weights, models, and caches present without network
demo-check:
	uv run python scripts/demo_check.py

# build the 300k-vector scale shard (press S in the UI to attach it)
demo-scale:
	uv run python scripts/preload_scale.py

# golden "yesterday's memory" state: teach on stage conditions once, then save
demo-save:
	rm -rf demo-golden && cp -R edge-data demo-golden
	@echo "golden state saved (commit demo-golden/ to version it)"

demo-restore:
	rm -rf edge-data && cp -R demo-golden edge-data
	@echo "stage state restored from demo-golden/"

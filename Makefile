# Fleet Memory — build/run targets (PLAN.md §7)

.PHONY: setup test smoke soak lint run run-b reset fleet-up fleet-down \
        demo-check demo-restore demo-save demo-scale

setup:
	uv sync --all-extras

test:
	uv run pytest tests/gate -q

smoke:
	uv run pytest tests/smoke -q

soak:
	uv run python scripts/soak.py --source 0 --minutes 10

lint:
	uv run ruff check .
	uv run ruff format --check .

fleet-up:
	docker compose up -d

fleet-down:
	docker compose down

run:
	uv run python -m fleetmemory.server

run-b:
	uv run python -m fleetmemory.server --instance b

reset:
	rm -rf edge-data

# --- demo rituals (PLAN.md §4) ---

demo-check:
	uv run python scripts/demo_check.py
	uv run pytest tests/smoke tests/drive -q
	uv run pytest tests/sync -q

demo-scale:
	uv run python scripts/preload_scale.py

# golden "yesterday's memory" state: teach on stage conditions once, then save
demo-save:
	rm -rf demo-golden && cp -R edge-data demo-golden
	@echo "golden state saved (commit demo-golden/ to version it)"

demo-restore:
	rm -rf edge-data && cp -R demo-golden edge-data
	@echo "stage state restored from demo-golden/"

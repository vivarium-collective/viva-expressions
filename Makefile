.PHONY: workbench-update
workbench-update:
	uv lock --upgrade-package vivarium-workbench
	uv sync --all-extras

.PHONY: workbench-serve-local
workbench-serve-local: workbench-update
	@uv run vivarium-workbench serve --workspace ./

.PHONY: workbench-serve-remote
workbench-serve-remote: workbench-update
	@uv run vivarium-workbench serve --workspace ./ --backend-base-url https://sms.cam.uchc.edu
.PHONY: workbench-update
workbench-update:
	uv lock --upgrade-package vivarium-workbench
	uv sync --all-extras

.PHONY: workbench-serve
workbench-serve:
	@uv run vivarium-workbench serve --workspace ./ --backend-base-url https://sms.cam.uchc.edu

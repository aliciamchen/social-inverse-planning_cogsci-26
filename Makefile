.PHONY: help fits cv values figures test all clean

PY := uv run python

EXPERIMENTS_FORW := food_forw_intimacy_desire
EXPERIMENTS_INV  := food_inv_intimacy_desire_alt food_inv_desire_intimacy_alt

QMDS := \
	analysis/food-forw-intimacy-desire-analysis.qmd \
	analysis/food-inv-intimacy-desire-alt-analysis.qmd \
	analysis/food-inv-desire-intimacy-alt-analysis.qmd \
	analysis/inv-plan-combined-correlation.qmd

help:
	@echo "Targets:"
	@echo "  fits        Run all forward + inverse fits"
	@echo "  cv          Run leave-one-scenario-out cross-validation"
	@echo "  values      Regenerate cogsci-2026/values.tex from latest fits"
	@echo "  figures     Render the 4 analysis qmds (regenerates figures/)"
	@echo "  test        Run model compliance tests"
	@echo "  all         fits + cv + values + figures"
	@echo "  clean       Remove generated outputs (model/outputs/, figures/, *.html)"

fits:
	$(PY) model/forward/fit_food_forw_intimacy_desire.py
	$(PY) model/inverse/fit_food_inv_intimacy_desire_alt.py
	$(PY) model/inverse/fit_food_inv_desire_intimacy_alt.py

cv:
	$(PY) model/cv/cv_food_forw_intimacy_desire.py
	$(PY) model/cv/cv_food_inv_intimacy_desire_alt.py
	$(PY) model/cv/cv_food_inv_desire_intimacy_alt.py

values:
	$(PY) scripts/generate_values_tex.py

figures:
	@for qmd in $(QMDS); do \
		quarto render $$qmd; \
	done

test:
	$(PY) model/test_model_compliance.py

all: fits cv values figures

clean:
	rm -rf model/outputs/
	rm -rf figures/
	rm -f analysis/*.html
	rm -rf analysis/*_files/
	rm -rf .quarto/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".DS_Store" -delete

.PHONY: data eval clean all

PYTHONPATH_EXPORT = PYTHONPATH=.

all: data eval

data:
	$(PYTHONPATH_EXPORT) python3 src/generate_synthetic_data.py
	$(PYTHONPATH_EXPORT) python3 src/data_prep.py --csv data/twcs_synthetic.csv --brand AmazonHelp --out data/cases.csv
	$(PYTHONPATH_EXPORT) python3 eval/build_golden_set.py
	$(PYTHONPATH_EXPORT) python3 eval/build_calibration_set.py

eval:
	$(PYTHONPATH_EXPORT) python3 eval/run_eval.py

clean:
	rm -f data/twcs_synthetic.csv data/ground_truth_intents.csv data/cases.csv
	rm -f eval/golden_set.csv eval/human_judge_calibration.csv eval/results_summary.csv
	rm -f eval/system_predictions.csv eval/system_judge_scores.csv eval/judge_agreement.json
	rm -f eval/system_per_class_metrics.csv eval/system_confusion_matrix.csv

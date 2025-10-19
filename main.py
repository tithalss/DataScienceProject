from scripts.analysis_script import run_analysis
from scripts.ml_metrics_script import run_model_evaluation
from scripts.ml_script import run_ml_analysis
from scripts.modeling_script import run_modeling
from scripts.preprocessing_script import run_preprocessing
from scripts.visualization_script import run_visualization

if __name__ == "__main__":
    run_preprocessing()
    run_analysis()
    run_modeling()
    run_ml_analysis()
    run_model_evaluation()
    run_visualization()


from concept_normalisation.pipeline.run import ExperimentConfig, run_pipeline


def run() -> None:
    run_pipeline(ExperimentConfig())


if __name__ == "__main__":
    run()

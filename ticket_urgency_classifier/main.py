"""Main CLI interface for the ticket urgency classifier."""

import typer

from ticket_urgency_classifier.dataset import app as dataset_app
from ticket_urgency_classifier.features import app as features_app
from ticket_urgency_classifier.modeling.evaluate import app as evaluate_app
from ticket_urgency_classifier.modeling.train import app as train_app
from ticket_urgency_classifier.validate import app as validate_app

app = typer.Typer()

# Register commands from other modules
app.add_typer(validate_app, name="validate")
app.add_typer(dataset_app, name="dataset")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")


@app.command()
def feature_pipeline():
    """Run the feature pipeline: validation -> dataset -> features"""
    # Run validation
    from ticket_urgency_classifier.validate import raw as validate_raw

    validate_raw()

    # Run dataset processing
    from ticket_urgency_classifier.dataset import main as dataset_main

    dataset_main()

    from ticket_urgency_classifier.features import main as features_main

    # Run feature engineering
    features_main()


@app.command()
def full_pipeline():
    """Run the full pipeline: raw validation -> dataset -> features -> processed validation -> prepare-model-data -> train -> evaluate."""
    # Run raw validation
    from ticket_urgency_classifier.validate import raw as validate_raw

    validate_raw()

    # Run dataset processing
    from ticket_urgency_classifier.dataset import main as dataset_main

    dataset_main()

    # Run feature engineering
    from ticket_urgency_classifier.features import main as features_main

    features_main()

    # Run processed data validation
    from ticket_urgency_classifier.validate import processed as validate_processed

    validate_processed()

    # Run data preparation for modeling
    from ticket_urgency_classifier.prepare_model_data import main as prepare_model_data_main

    prepare_model_data_main()

    # Run model training
    from ticket_urgency_classifier.modeling.train import main as train_main

    train_main()

    # Run model evaluation
    from ticket_urgency_classifier.modeling.evaluate import main as evaluate_main

    evaluate_main()


if __name__ == "__main__":
    app()

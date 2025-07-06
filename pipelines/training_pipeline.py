from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.filter_text_step import filter_text_step
from steps.feature_engineering_step import feature_engineering_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step

from zenml import Model, pipeline, step


@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="multi-label-news-categorization-classifier",
    ), 
    # enable_cache=False
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="data/archive.zip"
    )

    # Handling Missing Values Step
    filtered_data = filter_text_step(raw_data)
    
    # Feature Engineering Step
    engineered_data = feature_engineering_step(
        filtered_data, basic=True, stopword_removal=True, stemming=True, features=['content']
    )
    
    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(engineered_data)
    
    # Model Building Step
    trained_pipeline = model_building_step(X_train=X_train, 
                                           y_train=y_train, 
                                           # model_type='logistic_regression', params={})
                                           model_type='lightgbm',
                                           params={'learning_rate': 0.08, 
                                            'num_leaves': 35, 
                                            'n_estimators': 350, 
                                            'verbose':-1
                                            })

    # Model Evaluation Step
    evaluation_metrics = model_evaluator_step(
        trained_pipeline=trained_pipeline, X_test=X_test, y_test=y_test
    )
    print(evaluation_metrics)

    return trained_pipeline


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()

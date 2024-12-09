import pandas as pd
from autogluon.tabular import TabularPredictor
import warnings

warnings.filterwarnings('ignore')

processed_df = pd.read_csv("train_level_1.csv")
processed_df = processed_df.drop(columns=['DateTime'])

predictor = TabularPredictor(
    label='Power(mW)',
    eval_metric='mean_absolute_error',
    path='level_1',
    verbosity=2,
)

# Train the model
predictor.fit(
    train_data=processed_df,
    time_limit=8*60*60,
    presets='best_quality',
    excluded_model_types=['NN_TORCH', 'FASTAI', 'KNN', 'XGB', 'CAT'],
    keep_only_best=True,
    dynamic_stacking=False,
)
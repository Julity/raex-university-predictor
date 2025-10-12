#D:\Нейронка\RAEX_Project\src\preprocess.py
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Tuple, Optional

def scale_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
    else:
        scaled = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return scaled_df, scaler

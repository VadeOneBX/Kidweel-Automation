import pandas as pd
import joblib
import os

class SyntheticMLScorer:
    def __init__(self, model_dir="models"):
        self.model_path = os.path.join(model_dir, "synthetic_logistic_model.pkl")
        self.scaler_path = os.path.join(model_dir, "synthetic_scaler.pkl")
        self.is_trained = False
        
        self.features = ['net_delta', 'alpha', 'spot_in_vol_range', 'dist_to_high_vol', 'dist_to_low_vol']

        # Attempt to load ML artifacts
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
            print("✅ Loaded trained Synthetic ML Scorer.")
        else:
            print("⚠️ ML Artifacts not found. Run train_model.py first. Falling back to heuristic scoring.")

    def _engineer_features(self, candidates_df: pd.DataFrame, sg_row: pd.Series, spot: float) -> pd.DataFrame:
        df = candidates_df.copy()
        high_vol = sg_row.get('High Vol Point', spot)
        low_vol = sg_row.get('Low Vol Point', spot)
        
        range_width = high_vol - low_vol
        df['spot_in_vol_range'] = (spot - low_vol) / range_width if range_width > 0 else 0.5
        df['dist_to_high_vol'] = (df['short_strike'] - high_vol) / high_vol
        df['dist_to_low_vol'] = (df['short_strike'] - low_vol) / low_vol
        df['alpha'] = df['premium'] / (df['max_loss'] + df['premium']) # Width proxy
        return df

    def predict_win_probability(self, candidates_df: pd.DataFrame, sg_row: pd.Series, spot: float) -> pd.DataFrame:
        if candidates_df.empty:
            return candidates_df

        df = self._engineer_features(candidates_df, sg_row, spot)
        
        if self.is_trained:
            X = self.scaler.transform(df[self.features].fillna(0))
            df['ml_win_prob'] = self.model.predict_proba(X)[:, 1]
        else:
            # Fallback heuristic
            df['ml_win_prob'] = 1.0 - abs(df['net_delta'].abs() - 0.25)
            
        return df.sort_values('ml_win_prob', ascending=False)
from sklearn.calibration import CalibratedClassifierCV

class ProbabilityCalibrator:
    """Wraps a trained classifier with probability calibration."""

    def __init__(self, method='isotonic', cv='prefit'):
        self.method = method
        self.cv = cv
        self.calibrated_model = None

    def fit(self, model, X_valid, y_valid):
        # Model must be prefit if cv='prefit'
        self.calibrated_model = CalibratedClassifierCV(model, method=self.method, cv=self.cv)
        self.calibrated_model.fit(X_valid, y_valid)
        return self

    def predict_proba(self, X):
        if self.calibrated_model is None:
            raise ValueError("Calibrator not fitted")
        return self.calibrated_model.predict_proba(X) 
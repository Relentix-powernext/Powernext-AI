"""
CPRI Hackathon - Reproducible Pipeline (v2)
=============================================
Fix vs v1: the validity classifier's strongest feature must NOT require the
true Reference_Parameter (test data doesn't have it). Replaced with the
disagreement between a physics-only model (V, I, Ambient, Duration) and the
full model (+ S1, S2, S3) -- this is computable identically on train and test.

Run: python3 pipeline_v2.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, r2_score, roc_auc_score

RANDOM_STATE = 42
SENS_COLS = ['Sensor_S1', 'Sensor_S2', 'Sensor_S3']
OP_COLS = ['Applied_Voltage_kV', 'Load_Current_A', 'Ambient_Temperature_C', 'Test_Duration_min']
FEAT_COLS = OP_COLS + SENS_COLS
ALL_INPUT_COLS = OP_COLS + SENS_COLS + ['Sensor_S4']
SENTINEL_VALUES = {0, 1, 25}

CLF_FEATURES = ['n_sentinel', 'n_missing_sensor', 's4_missing', 'is_duplicate',
                's3_le_s1', 'max_abs_sensor_z', 'sensor_max_abs_dev',
                'sensor_spread_ratio', 's1_s3_gap', 'disagreement_pct']


# ---------------------------------------------------------------------------
# Feature engineering -- identical code path for train and test
# ---------------------------------------------------------------------------
def null_sentinels(df):
    d = df.copy()
    sentinel_mask = pd.DataFrame(False, index=d.index, columns=SENS_COLS)
    for c in SENS_COLS:
        sentinel_mask[c] = d[c].isin(SENTINEL_VALUES) | (d[c] < 0)
        d.loc[sentinel_mask[c], c] = np.nan
    d['n_sentinel'] = sentinel_mask.sum(axis=1)
    return d


def add_duplicate_flag(d):
    d['dup_group_size'] = d.groupby(ALL_INPUT_COLS)[ALL_INPUT_COLS[0]].transform('size')
    d['is_duplicate'] = (d['dup_group_size'] > 1).astype(int)
    return d


def add_missing_flags(d):
    d['n_missing_sensor'] = d[SENS_COLS].isna().sum(axis=1)
    d['s4_missing'] = d['Sensor_S4'].isna().astype(int)
    return d


def add_ordering_flag(d):
    d['s3_le_s1'] = ((d['Sensor_S3'] <= d['Sensor_S1'])
                      & d['Sensor_S1'].notna() & d['Sensor_S3'].notna()).astype(int)
    return d


def add_sensor_consistency_features(d):
    med = d[SENS_COLS].median(axis=1)
    dev = d[SENS_COLS].sub(med, axis=0)
    d['sensor_max_abs_dev'] = dev.abs().max(axis=1)
    d['sensor_spread_ratio'] = d['sensor_max_abs_dev'] / (med.abs() + 1e-6)
    d['s1_s3_gap'] = (d['Sensor_S1'] - d['Sensor_S3']).abs()
    return d


def fit_sensor_expectation_models(clean_valid_df):
    models = {}
    for s in SENS_COLS:
        lr = LinearRegression().fit(
            clean_valid_df[['Load_Current_A', 'Ambient_Temperature_C']], clean_valid_df[s])
        resid_std = (clean_valid_df[s] - lr.predict(
            clean_valid_df[['Load_Current_A', 'Ambient_Temperature_C']])).std()
        models[s] = (lr, resid_std)
    return models


def apply_sensor_expectation_models(d, models):
    for s, (lr, std) in models.items():
        pred = lr.predict(d[['Load_Current_A', 'Ambient_Temperature_C']])
        d[s + '_z'] = (d[s] - pred) / std
    d['max_abs_sensor_z'] = d[[s + '_z' for s in SENS_COLS]].abs().max(axis=1)
    return d


def build_features(df, sensor_impute_values, rf_physics, rf_full, sensor_exp_models):
    """Ground-truth-free feature pipeline -- works identically on train and test."""
    d = null_sentinels(df)
    d = add_duplicate_flag(d)
    d = add_missing_flags(d)
    d = add_ordering_flag(d)
    d = add_sensor_consistency_features(d)
    d = apply_sensor_expectation_models(d, sensor_exp_models)

    d_filled = d.copy()
    for c in SENS_COLS:
        d_filled[c] = d_filled[c].fillna(sensor_impute_values[c])

    d['pred_physics'] = rf_physics.predict(d_filled[OP_COLS])
    d['pred_full'] = rf_full.predict(d_filled[FEAT_COLS])
    d['disagreement_pct'] = ((d['pred_full'] - d['pred_physics']) / d['pred_physics'] * 100).abs()
    d['Predicted_Reference_Parameter'] = d['pred_full']
    return d, d_filled


def main():
    report = []
    def log(*a):
        line = ' '.join(str(x) for x in a)
        print(line); report.append(line)

    xl = pd.ExcelFile('/mnt/user-data/uploads/CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx')
    train = xl.parse('Training_Data')
    test = xl.parse('Test_Data')

    log("="*70); log("CPRI PIPELINE v2 - TRAINING PHASE"); log("="*70)
    log(f"Training rows: {len(train)}   Test rows: {len(test)}")

    # ---- fit everything on TRAIN only ----
    train_null = null_sentinels(train)
    train_null = add_duplicate_flag(train_null)
    clean_valid_mask = ((train_null.Validity_Label == 'Valid')
                         & (train_null.n_sentinel == 0)
                         & train_null[SENS_COLS].notna().all(axis=1)
                         & (train_null.is_duplicate == 0))
    clean_valid = train_null[clean_valid_mask]
    log(f"Clean Valid rows used to fit baseline models: {len(clean_valid)}")

    sensor_impute_values = clean_valid[SENS_COLS].median()
    sensor_exp_models = fit_sensor_expectation_models(clean_valid)

    rf_physics = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, min_samples_leaf=2)
    rf_physics.fit(clean_valid[OP_COLS], clean_valid['Reference_Parameter'])
    rf_full = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, min_samples_leaf=2)
    rf_full.fit(clean_valid[FEAT_COLS], clean_valid['Reference_Parameter'])

    kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = cross_val_predict(RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE,
                                  min_samples_leaf=2), clean_valid[FEAT_COLS],
                                  clean_valid['Reference_Parameter'], cv=kf)
    r2 = r2_score(clean_valid['Reference_Parameter'], oof_pred)
    log(f"\nReference_Parameter regression - 5-fold CV R^2 (clean Valid rows): {r2:.4f}")

    imp = pd.Series(rf_full.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    log("\nFeature importance (Reference_Parameter regressor):")
    for k, v in imp.items(): log(f"  {k:25s} {v:.4f}")

    # ---- build classifier features on TRAIN ----
    train_feat, train_filled = build_features(train, sensor_impute_values, rf_physics, rf_full, sensor_exp_models)
    y = (train_feat.Validity_Label == 'Invalid').astype(int)
    X = train_feat[CLF_FEATURES]

    log(f"\nAUC of disagreement_pct alone (ground-truth-free residual proxy): "
        f"{roc_auc_score(y, train_feat['disagreement_pct']):.4f}")

    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE,
                                  class_weight='balanced', min_samples_leaf=2)
    oof_proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]
    oof_label = (oof_proba >= 0.5).astype(int)
    log("\nValidity classifier - 5-fold CV performance (ALL features ground-truth-free):")
    log(classification_report(y, oof_label, target_names=['Valid', 'Invalid']))
    log(f"AUC: {roc_auc_score(y, oof_proba):.4f}")

    clf.fit(X, y)
    imp_clf = pd.Series(clf.feature_importances_, index=CLF_FEATURES).sort_values(ascending=False)
    log("Feature importance (Validity classifier):")
    for k, v in imp_clf.items(): log(f"  {k:25s} {v:.4f}")

    # ---- apply identical fitted pipeline to TEST_DATA ----
    log("\n" + "="*70); log("APPLYING PIPELINE TO TEST_DATA (350 rows, transform-only)"); log("="*70)
    test_feat, test_filled = build_features(test, sensor_impute_values, rf_physics, rf_full, sensor_exp_models)
    X_test = test_feat[CLF_FEATURES]
    test_proba = clf.predict_proba(X_test)[:, 1]
    test_feat['Validity_Label'] = np.where(test_proba >= 0.5, 'Invalid', 'Valid')
    test_feat['invalid_probability'] = test_proba

    n_invalid = (test_feat.Validity_Label == 'Invalid').sum()
    log(f"\nTest_Data validity predictions: {n_invalid} Invalid / {len(test_feat)-n_invalid} Valid "
        f"({n_invalid/len(test_feat)*100:.1f}% Invalid, vs {y.mean()*100:.1f}% in training)")

    log("\nFault-signature breakdown on Test_Data:")
    log(f"  Exact duplicate-conflict candidates : {(test_feat.is_duplicate==1).sum()}")
    log(f"  Sentinel sensor values              : {(test_feat.n_sentinel>0).sum()}")
    log(f"  Missing critical sensor              : {(test_feat.n_missing_sensor>0).sum()}")
    log(f"  S3<=S1 ordering violation            : {(test_feat.s3_le_s1==1).sum()}")
    log(f"  Sensor-vs-current outlier (z>3)      : {(test_feat.max_abs_sensor_z>3).sum()}")
    log(f"  High physics/sensor disagreement (>8%): {(test_feat.disagreement_pct>8).sum()}")

    # sanity: does every Invalid prediction trace to >=1 structural reason, or is it purely model judgment?
    structural_any = ((test_feat.is_duplicate==1)|(test_feat.n_sentinel>0)|(test_feat.n_missing_sensor>0)
                       |(test_feat.s3_le_s1==1)|(test_feat.max_abs_sensor_z>3)|(test_feat.disagreement_pct>8))
    pred_invalid = test_feat.Validity_Label=='Invalid'
    log(f"\nPredicted-Invalid rows WITH a traceable structural reason: "
        f"{(pred_invalid & structural_any).sum()}/{pred_invalid.sum()}")
    log(f"Predicted-Invalid rows from classifier judgment ALONE (no single rule fired): "
        f"{(pred_invalid & ~structural_any).sum()}/{pred_invalid.sum()}")

    submission = test_feat[['Test_ID', 'Predicted_Reference_Parameter', 'Validity_Label']].copy()
    submission['Predicted_Reference_Parameter'] = submission['Predicted_Reference_Parameter'].round(4)
    submission.to_csv('/home/claude/cpri/submission.csv', index=False)

    diag_cols = (['Test_ID'] + ALL_INPUT_COLS +
                 ['Predicted_Reference_Parameter', 'Validity_Label', 'invalid_probability',
                  'n_sentinel', 'n_missing_sensor', 'is_duplicate', 's3_le_s1',
                  'max_abs_sensor_z', 'disagreement_pct'])
    test_feat[diag_cols].to_csv('/home/claude/cpri/test_predictions_diagnostic.csv', index=False)

    with open('/home/claude/cpri/pipeline_report.txt', 'w') as f:
        f.write('\n'.join(report))
    log("\nDone.")


if __name__ == '__main__':
    main()

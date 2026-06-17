import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Engagement'] = df['Pages_Viewed'] + df['Products_Viewed']
    df['Time_per_Product'] = df['Time_On_Site'] / (df['Products_Viewed'] + 1)
    df['Income_per_Page'] = df['Income'] / (df['Pages_Viewed'] + 1)
    df['High_Income'] = (df['Income'] > df['Income'].median()).astype(int)
    df['Returns_to_Site'] = df['Previous_Purchases'] > 0
    return df


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    public = pd.read_csv('public_test.csv')
    private = pd.read_csv('private_test.csv')

    # Combine labeled data for final training
    public_eval = public.copy()
    full_train = pd.concat([train, public_eval], ignore_index=True)

    # Feature engineering
    train = add_features(train)
    public = add_features(public)
    private = add_features(private)

    categorical_cols = ['Device_Type', 'Traffic_Source']
    numeric_cols = [
        'Age', 'Income', 'Pages_Viewed', 'Products_Viewed', 'Time_On_Site',
        'Previous_Purchases', 'City_Tier', 'Discount_Seen', 'Browser_Version',
        'Campaign_Code', 'Engagement', 'Time_per_Product', 'Income_per_Page',
        'High_Income', 'Returns_to_Site'
    ]

    preprocess = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical_cols),
    ], remainder='drop')

    model = Pipeline([
        ('preprocess', preprocess),
        ('clf', HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=0.1))
    ])

    X_train = train[numeric_cols + categorical_cols]
    y_train = train['Converted']

    # Cross-validation on train set for baseline assessment
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    print('Train CV F1 scores:', scores)
    print('Train CV F1 mean:', scores.mean())

    # Evaluate on public test as holdout
    X_public = public[numeric_cols + categorical_cols]
    y_public = public['Converted']
    model.fit(X_train, y_train)
    public_pred = model.predict(X_public)
    print('Public holdout F1:', f1_score(y_public, public_pred))

    # Final model on all labeled data
    combined = pd.concat([train, public], ignore_index=True)
    X_combined = combined[numeric_cols + categorical_cols]
    y_combined = combined['Converted']
    model.fit(X_combined, y_combined)

    X_private = private[numeric_cols + categorical_cols]
    private_preds = model.predict(X_private)

    submission = pd.DataFrame({'User_ID': private['User_ID'], 'Converted': private_preds.astype(int)})
    submission.to_csv('submission.csv', index=False)
    print('submission.csv written with', submission.shape[0], 'rows')

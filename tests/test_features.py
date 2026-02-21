from api.features import extract_fake_features, fake_score

def test_fake_review_scores_high():
    fake = 'AMAZING!!! BEST PRODUCT EVER!!! HIGHLY RECOMMEND!!!'
    features = extract_fake_features(fake)
    score = fake_score(features)
    assert score >= 0.5, f'Expected >= 0.5, got {score}'

def test_genuine_review_scores_low():
    genuine = 'Battery life lasts about 8 hours. Screen resolution is decent but not great outdoors.'
    features = extract_fake_features(genuine)
    score = fake_score(features)
    assert score < 0.5, f'Expected < 0.5, got {score}'
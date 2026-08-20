from ai.recommendation.season import get_ranked_season_recommendations

# Use the actual values from your face_result
face_result = {
    'depth': 'deep',
    'undertone': 'cool',
    'clarity': 'muted',
    'contrast': 'medium'  # or whatever your actual contrast was
}

ranked = get_ranked_season_recommendations(face_result)
print("Winner:", ranked['top_seasons'][0]['season_label'])
print("Score:", ranked['top_seasons'][0]['score_pct'])
# Calculate the predicted vote share for each party in a given constituency
def calculate_predicted_vote_share(national_forecast: dict, bias_scores: pd.DataFrame):
    '''
    Input >
    national_forecast: a dictionary with the national forecasted vote share for each party
    bias_scores: a dataframe with the bias score for each party in a given constituency.

    This function takes the national vote share forecast and the bias scores for each party in a given constituency,
    combines them to calculate the predicted vote share for each party in that constituency.

    Output > predicted_vote_share: a dictionary with the predicted vote share for each party in the constituency.
    '''
    predicted_vote_share = {}
    for party, forecast_share in national_forecast.items():
        bias_score = bias_scores[party + '_Bias_Score']
        predicted_vote_share[party] = forecast_share + bias_score
    return predicted_vote_share

# Function to determine the winning party in a constituency
def determine_winner(predicted_vote_share: dict):
    '''
    Input > predicted_vote_share: a dictionary with the predicted vote share for each party in the constituency.

    This function takes the predicted vote share for each party [calculate_predicted_vote_share() output] in a given constituency and returns the party with the highest vote share.

    Output > winner: the party with the highest predicted vote share for a given constituency.
    '''
    return max(predicted_vote_share, key=predicted_vote_share.get)

# Determine the winner for all constituencies
def determine_winners(national_forecast, test_bias):
    '''
    Input >
    national_forecast: a dictionary with the national forecasted vote share for each party
    test_bias: a dataframe with the bias score for each party in each constituency.

    This function takes the national vote share forecast and the bias scores for each party in each constituency,

    Output > winners: a dataframe with the winning party for each constituency.
    '''
    winners = []
    for index, row in test_bias.iterrows():
        predicted_vote_share = calculate_predicted_vote_share(national_forecast, row)
        winner = determine_winner(predicted_vote_share)
        winners.append({'Constituency': row['Constituency'], 'Winner': winner})
    return pd.DataFrame(winners)

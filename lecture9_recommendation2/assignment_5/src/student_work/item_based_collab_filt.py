import numpy as np



def get_item_based_rating_prediction(target_user=None, target_item=None, user_item_ratings_df=None,
                                     item_item_similarity_dictionary=None, neighborhood_similarity_threshold=None):

    # get the ratings of the target user for all items
    target_user_ratings = user_item_ratings_df.loc[target_user, :]
#     print(target_user_ratings)
    # get the ratings of all users for the target item
    item_ratings = user_item_ratings_df[target_item]
    # get the users who have rated the target item and have non-NaN ratings for target_user's rated items
    candidate_users = item_ratings[(item_ratings.notna()) & (target_user_ratings.notna())].index.tolist()
    
    # filter out users with similarity below threshold
#     candidate_similarities = [(item_item_similarity_dictionary[(target_item, item)], item_ratings.loc[item]) 
#                               for item in candidate_users 
#                               if (target_item, item) in item_item_similarity_dictionary or (item,target_item) in item_item_similarity_dictionary]
    candidate_similarities = []
    for item in candidate_users:
        if (target_item, item) in item_item_similarity_dictionary or(item,target_item) in  item_item_similarity_dictionary:
            if (target_item, item) in item_item_similarity_dictionary:
                similarity = item_item_similarity_dictionary[(target_item, item)]
            else:
                similarity=item_item_similarity_dictionary[(item,target_item)]
            rating = item_ratings.loc[item]
            candidate_similarities.append((similarity, rating))
    # print(candidate_similarities)
    candidate_similarities = [(similarity, rating) 
                              for similarity, rating in candidate_similarities 
                              if similarity > neighborhood_similarity_threshold]
    # print(candidate_similarities)
    
    if not candidate_similarities:
        return item_ratings.mean()
    
    # compute weighted average of the ratings of the target item by similar users
    numerator = sum(similarity * (rating - item_ratings.mean()) 
                    for similarity, rating in candidate_similarities)
    denominator = sum(abs(similarity) for similarity, _ in candidate_similarities)

    item_based_rating_prediction = user_item_ratings_df.loc[target_user, :].mean() + (numerator / denominator)
    
    return item_based_rating_prediction


def get_item_ratings_for_target_user(target_user=None, user_item_ratings_df=None, item_item_similarity_dictionary=None,
                                     neighborhood_similarity_threshold=None):
    """
    In this function you will generate a dictionary of predicted ratings for the target user for all items that
    the target user has not rated.
    The dictionary keys will be the item id as a string.
    The dictionary values will be the predicted ratings as a float.
    Once the dictionary is completed sort the dictionary by predicted rating in decending order.
    """

    pred_dict = {}

    for item in user_item_ratings_df.columns:
        # skip items that have already been rated by the target user
        if not np.isnan(user_item_ratings_df.loc[target_user, item]):
            continue

        # get predicted rating for item
        pred_rating = get_item_based_rating_prediction(target_user, item, user_item_ratings_df,
                                                        item_item_similarity_dictionary,
                                                        neighborhood_similarity_threshold)
        # print(pred_rating)
        # add predicted rating to dictionary
        if not np.isnan(pred_rating):
            pred_dict[str(item)] = pred_rating

    # sort dictionary by predicted rating in descending order
    pred_dict = {k: v for k, v in sorted(pred_dict.items(), key=lambda item: item[1], reverse=True)}

    return pred_dict


if __name__ == '__main__':
    pass

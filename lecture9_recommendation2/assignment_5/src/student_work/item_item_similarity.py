import numpy as np
def get_table_2_1_item_item_similarity_dictionary():

    table_2_1_item_item_sim_dict = {
        ('1', '2'): 0.7350831879156821,
        ('1', '3'): 0.9116846116771036,
        ('1', '4'): -0.8483022651467478,
        ('1', '5'): -0.8124881033019942,
        ('1', '6'): -0.989620304187242,
        ('2', '3'): 0.8728715609439694,
        ('2', '4'): -0.7339104058141734,
        ('2', '5'): -0.9959988636406356,
        ('2', '6'): -0.6222507260998813,
        ('3', '4'): -0.8819171036881968,
        ('3', '5'): -0.8944271909999157,
        ('3', '6'): -0.9116846116771036,
        ('4', '5'): 0.7056710903752627,
        ('4', '6'): 0.8289958835741487,
        ('5', '6'): 0.7303362637517609
    }

    return table_2_1_item_item_sim_dict


def get_item_item_similarity_dictionary(a_user_item_ratings_matrix_df=None):
    """
    Reference: https://link.springer.com/book/10.1007/978-3-319-29659-3 - Charu C. Aggarwal, Recommender Systems - The
    Textbook (free for download)

    This function will return an item-item dictionary.

    The dictionary keys will be a tuple whose elements identify the two items forming the similarity. The elements will
    be strings sorted in ascending order.

    The dictionary values will be the adjusted cosine similarity as calculated in equation 2.14 or some other method.

    :return:
    """

    def adjusted_cosine_similarity(item1, item2):
        # Get the set of users who have rated both items
        users_set = set(a_user_item_ratings_matrix_df[item1].dropna().index) & set(a_user_item_ratings_matrix_df[item2].dropna().index)

        # Compute the adjusted cosine similarity between the two items
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        for user in users_set:
            rating1 = a_user_item_ratings_matrix_df.loc[user, item1]
            rating2 = a_user_item_ratings_matrix_df.loc[user, item2]
            avg_rating = a_user_item_ratings_matrix_df.loc[user].mean()

            numerator += (rating1 - avg_rating) * (rating2 - avg_rating)
            denominator1 += (rating1 - avg_rating)**2
            denominator2 += (rating2 - avg_rating)**2

        if denominator1 == 0 or denominator2 == 0:
            return 0

        return numerator / (np.sqrt(denominator1) * np.sqrt(denominator2))

    # Initialize an empty dictionary to hold the item-item similarities
    item_item_similarity_dictionary = {}

    # Iterate over all pairs of items in the rating matrix
    for item1 in a_user_item_ratings_matrix_df.columns:
        for item2 in a_user_item_ratings_matrix_df.columns:
            if item1 == item2:
                continue

            # Sort the items in ascending order and use them as the key for the dictionary
            key = tuple(sorted([item1, item2]))

            # If the key is already in the dictionary, continue to the next pair of items
            if key in item_item_similarity_dictionary:
                continue

            # Calculate the adjusted cosine similarity between the two items
            similarity = adjusted_cosine_similarity(item1, item2)

            # Add the similarity to the dictionary
            item_item_similarity_dictionary[key] = similarity

    return item_item_similarity_dictionary

if __name__ == '__main__':
    pass

